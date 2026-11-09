// OpenFilm's OpenFX entry point.
//
// How Resolve talks to a plugin, in short:
//  1. On startup it dlopen()s OpenFilm.ofx and calls OfxGetPlugin() (provided by the OFX Support
//     library, which asks us via OFX::Plugin::getPluginIDs below).
//  2. "describe" / "describeInContext": we tell Resolve our name, which image formats we accept,
//     that we can render with Metal, and which knobs (parameters) to show.
//  3. "createInstance": a node using OpenFilm was created; we make an OpenFilmEffect for it.
//  4. "render": per frame, Resolve hands us the source and destination images plus the knob values,
//     either as CPU memory or (with Metal on) as GPU buffers. We forward to the matching renderer.
//
// All film maths lives in src/core (CPU) and src/metal (GPU). This file only translates OFX.
#include <cmath>

#include "ofxsImageEffect.h"

#include "core/film_renderer.hpp"
#include "core/lut.hpp"
#include "metal/metal_renderer.h"

#include <dlfcn.h>

#include <algorithm>
#include <exception>
#include <stdexcept>
#include <memory>
#include <string>
#include <vector>

namespace {

constexpr const char* kPluginIdentifier = "org.openfilm.OpenFilm";
constexpr const char* kPluginName = "OpenFilm";
constexpr const char* kParamColourGroup = "colour";
constexpr const char* kParamInputSpace = "inputSpace";
constexpr const char* kParamOutputSpace = "outputSpace";
constexpr const char* kParamGamutCompression = "gamutCompression";
constexpr const char* kParamNegativeGroup = "negative";
constexpr const char* kParamNegativeStock = "negativeStock";
constexpr const char* kParamExposure = "exposure";
constexpr const char* kParamPrintGroup = "print";
constexpr const char* kParamPrintStock = "printStock";
constexpr const char* kParamPrinterLight[3] = {"printerLightR", "printerLightG", "printerLightB"};
constexpr const char* kParamPrintDensity = "printDensity";

// Path of this bundle's Resources folder, found from where the plugin binary was loaded:
// .../OpenFilm.ofx.bundle/Contents/MacOS/OpenFilm.ofx -> .../OpenFilm.ofx.bundle/Contents/Resources
std::string bundleResourcesDir() {
    Dl_info info{};
    if (!dladdr(reinterpret_cast<const void*>(&bundleResourcesDir), &info) || !info.dli_fname) return {};
    std::string path = info.dli_fname;
    path = path.substr(0, path.rfind('/'));  // .../Contents/MacOS
    path = path.substr(0, path.rfind('/'));  // .../Contents
    return path + "/Resources";
}

// One MetalRenderer for the whole process, created the first time a Metal render happens.
// (A function-local static is initialised exactly once, even with several render threads.)
openfilm::MetalRenderer& metalRenderer() {
    static openfilm::MetalRenderer renderer(bundleResourcesDir() + "/OpenFilm.metallib");
    return renderer;
}

// Film data (stocks, spectral tables) is read once per process. On failure `error` says why and
// `resources` stays null, so the plugin can describe itself and report the problem on render.
struct LoadedFilm {
    std::shared_ptr<const openfilm::FilmResources> resources;
    std::string error;
    std::unique_ptr<openfilm::FilmCache> cache;
    std::vector<const openfilm::Stock*> negatives, prints;  // choice index -> stock (id order)
};

const LoadedFilm& loadedFilm() {
    static const LoadedFilm film = [] {
        LoadedFilm f;
        try {
            f.resources = openfilm::FilmResources::load(bundleResourcesDir());
            f.cache = std::make_unique<openfilm::FilmCache>(f.resources);
            for (const auto& st : f.resources->stocks)
                (st.type == openfilm::StockType::Negative ? f.negatives : f.prints).push_back(&st);
            if (f.negatives.empty() || f.prints.empty()) throw std::runtime_error("no negative or print stock found");
        } catch (const std::exception& e) {
            f = LoadedFilm{};
            f.error = e.what();
        }
        return f;
    }();
    return film;
}

openfilm::ImageGeometry geometryOf(const OFX::Image& img) {
    const OfxRectI b = img.getBounds();
    return {b.x1, b.y1, b.x2, b.y2, std::ptrdiff_t(img.getRowBytes())};
}

bool isFloatRGBA(const OFX::Image& img) {
    return img.getPixelDepth() == OFX::eBitDepthFloat && img.getPixelComponents() == OFX::ePixelComponentRGBA;
}

class OpenFilmEffect : public OFX::ImageEffect {
public:
    explicit OpenFilmEffect(OfxImageEffectHandle handle) : ImageEffect(handle) {
        dstClip_ = fetchClip(kOfxImageEffectOutputClipName);
        srcClip_ = fetchClip(kOfxImageEffectSimpleSourceClipName);
        exposure_ = fetchDoubleParam(kParamExposure);
        gamutCompression_ = fetchDoubleParam(kParamGamutCompression);
        negativeStock_ = fetchChoiceParam(kParamNegativeStock);
        printStock_ = fetchChoiceParam(kParamPrintStock);
        for (int i = 0; i < 3; ++i) printerLight_[i] = fetchDoubleParam(kParamPrinterLight[i]);
        printDensity_ = fetchDoubleParam(kParamPrintDensity);
    }

    void render(const OFX::RenderArguments& args) override {
        std::unique_ptr<OFX::Image> dst(dstClip_->fetchImage(args.time));
        std::unique_ptr<OFX::Image> src(srcClip_->fetchImage(args.time));
        if (!dst || !src) OFX::throwSuiteStatusException(kOfxStatFailed);
        if (!isFloatRGBA(*dst) || !isFloatRGBA(*src)) OFX::throwSuiteStatusException(kOfxStatErrImageFormat);

        const openfilm::ImageGeometry srcGeom = geometryOf(*src);
        const openfilm::ImageGeometry dstGeom = geometryOf(*dst);
        const openfilm::RenderWindow win{args.renderWindow.x1, args.renderWindow.y1, args.renderWindow.x2,
                                         args.renderWindow.y2};
        const LoadedFilm& film = loadedFilm();
        if (!film.cache) {
            showError("OpenFilm: film data failed to load: " + film.error);
            OFX::throwSuiteStatusException(kOfxStatFailed);
        }
        openfilm::FilmParams fp;
        openfilm::LookControls controls;  // applied per pixel, never rebaked
        readLook(args, film, fp, controls);
        std::shared_ptr<const openfilm::BakedFilm> baked;
        try {
            baked = film.cache->get(fp);  // re-bakes (~1.4 s) only when the look changed
        } catch (const std::exception& e) {
            showError(std::string("OpenFilm: render failed: ") + e.what());
            OFX::throwSuiteStatusException(kOfxStatFailed);
        }

        if (args.isEnabledMetalRender) {
            // GPU path: getPixelData() is an id<MTLBuffer>, not CPU memory.
            openfilm::MetalRenderer& gpu = metalRenderer();
            if (!gpu.isValid()) {
                showError("OpenFilm: GPU shaders failed to load: " + gpu.error());
                OFX::throwSuiteStatusException(kOfxStatFailed);
            }
            if (!gpu.render(args.pMetalCmdQ, src->getPixelData(), srcGeom, dst->getPixelData(), dstGeom, win, baked,
                            controls)) {
                showError("OpenFilm: GPU render failed");
                OFX::throwSuiteStatusException(kOfxStatFailed);
            }
            clearError();
            return;
        }

        try {
            openfilm::renderFilmCpu(openfilm::ConstImageView{static_cast<const float*>(src->getPixelData()), srcGeom},
                                    openfilm::ImageView{static_cast<float*>(dst->getPixelData()), dstGeom}, win,
                                    *baked, controls);
        } catch (const std::exception& e) {
            showError(std::string("OpenFilm: render failed: ") + e.what());
            OFX::throwSuiteStatusException(kOfxStatFailed);
        }
        clearError();
    }

private:
    // The single place the controls are read, so the CPU and GPU paths can never disagree.
    void readLook(const OFX::RenderArguments& args, const LoadedFilm& film, openfilm::FilmParams& fp,
                  openfilm::LookControls& controls) {
        // Choice entries are listed by display name; the model wants stock ids.
        const auto pick = [&](const std::vector<const openfilm::Stock*>& list, OFX::ChoiceParam* p) {
            int i = 0;
            p->getValueAtTime(args.time, i);
            return list[size_t(std::clamp(i, 0, int(list.size()) - 1))]->id;
        };
        fp.negativeId = pick(film.negatives, negativeStock_);
        fp.printId = pick(film.prints, printStock_);
        fp.gamutCompression = gamutCompression_->getValueAtTime(args.time);
        controls.exposureStops = exposure_->getValueAtTime(args.time);
        for (int i = 0; i < 3; ++i) controls.printerPoints[size_t(i)] = printerLight_[i]->getValueAtTime(args.time);
        controls.printDensity = printDensity_->getValueAtTime(args.time);
    }

    // Resolve does not implement the OFX message suite's persistent messages; the Support library
    // turns that into a kOfxStatErrUnsupported exception, which would fail an otherwise finished
    // render (Resolve then discards the frame). Messages are best-effort, never fatal.
    void showError(const std::string& text) noexcept {
        try {
            setPersistentMessage(OFX::Message::eMessageError, "", text);
        } catch (...) {
        }
    }
    void clearError() noexcept {
        try {
            clearPersistentMessage();
        } catch (...) {
        }
    }

    OFX::Clip* dstClip_ = nullptr;
    OFX::Clip* srcClip_ = nullptr;
    OFX::DoubleParam* exposure_ = nullptr;
    OFX::DoubleParam* gamutCompression_ = nullptr;
    OFX::ChoiceParam* negativeStock_ = nullptr;
    OFX::ChoiceParam* printStock_ = nullptr;
    OFX::DoubleParam* printerLight_[3] = {};
    OFX::DoubleParam* printDensity_ = nullptr;
};

}  // namespace

mDeclarePluginFactory(OpenFilmFactory, {}, {});

void OpenFilmFactory::describe(OFX::ImageEffectDescriptor& desc) {
    desc.setLabels(kPluginName, kPluginName, kPluginName);
    desc.setPluginGrouping(kPluginName);
    desc.setPluginDescription("Physically based motion-picture film emulation (negative to print).");

    desc.addSupportedContext(OFX::eContextFilter);
    desc.addSupportedContext(OFX::eContextGeneral);
    desc.addSupportedBitDepth(OFX::eBitDepthFloat);  // Resolve converts everything else for us

    desc.setSingleInstance(false);
    desc.setHostFrameThreading(false);
    desc.setSupportsTiles(false);
    desc.setSupportsMultiResolution(false);
    desc.setTemporalClipAccess(false);
    desc.setRenderThreadSafety(OFX::eRenderFullySafe);
    // Resolve hands us GPU buffers and a command queue; the CPU path stays as the reference/fallback.
    desc.setSupportsMetalRender(true);
}

void OpenFilmFactory::describeInContext(OFX::ImageEffectDescriptor& desc, OFX::ContextEnum /*context*/) {
    OFX::ClipDescriptor* src = desc.defineClip(kOfxImageEffectSimpleSourceClipName);
    src->addSupportedComponent(OFX::ePixelComponentRGBA);
    src->setSupportsTiles(false);

    OFX::ClipDescriptor* dst = desc.defineClip(kOfxImageEffectOutputClipName);
    dst->addSupportedComponent(OFX::ePixelComponentRGBA);
    dst->setSupportsTiles(false);

    // Fills a stock drop-down with display names in id order (the same order render() maps back).
    const LoadedFilm& film = loadedFilm();
    const auto stockChoice = [&](const char* name, const char* label, const std::vector<const openfilm::Stock*>& list,
                                 OFX::GroupParamDescriptor* group) {
        OFX::ChoiceParamDescriptor* c = desc.defineChoiceParam(name);
        c->setLabel(label);
        if (list.empty()) c->appendOption("(stock data missing)");
        for (const auto* st : list) c->appendOption(st->displayName);
        c->setDefault(0);
        c->setParent(*group);
    };
    const auto doubleParam = [&](const char* name, const char* label, const char* hint, double def, double lo,
                                 double hi, double dispLo, double dispHi, double inc, OFX::GroupParamDescriptor* g) {
        OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(name);
        p->setLabel(label);
        p->setHint(hint);
        p->setDefault(def);
        p->setRange(lo, hi);
        p->setDisplayRange(dispLo, dispHi);
        p->setIncrement(inc);
        p->setDoubleType(OFX::eDoubleTypePlain);
        p->setParent(*g);
    };

    OFX::GroupParamDescriptor* colour = desc.defineGroupParam(kParamColourGroup);
    colour->setLabel("Colour");
    colour->setOpen(true);
    OFX::ChoiceParamDescriptor* in = desc.defineChoiceParam(kParamInputSpace);
    in->setLabel("Input space");
    in->appendOption("DaVinci Wide Gamut / Intermediate");
    in->setParent(*colour);
    OFX::ChoiceParamDescriptor* out = desc.defineChoiceParam(kParamOutputSpace);
    out->setLabel("Output space");
    out->appendOption("Rec.709 Gamma 2.4");
    out->setParent(*colour);
    doubleParam(kParamGamutCompression, "Gamut compression",
                "Pulls very saturated colours inside the gamut the film can reproduce (0 = hard clip at the gamut edge, 1 = full soft compression)", 0.5, 0.0,
                1.0, 0.0, 1.0, 0.05, colour);

    OFX::GroupParamDescriptor* negative = desc.defineGroupParam(kParamNegativeGroup);
    negative->setLabel("Negative");
    negative->setOpen(true);
    stockChoice(kParamNegativeStock, "Negative stock", film.negatives, negative);
    doubleParam(kParamExposure, "Exposure", "Negative exposure in stops", 0.0, -10.0, 10.0, -5.0, 5.0, 0.1, negative);

    OFX::GroupParamDescriptor* print = desc.defineGroupParam(kParamPrintGroup);
    print->setLabel("Print");
    print->setOpen(true);
    stockChoice(kParamPrintStock, "Print stock", film.prints, print);
    static const char* kLightLabel[3] = {"Printer light R", "Printer light G", "Printer light B"};
    for (int i = 0; i < 3; ++i)
        doubleParam(kParamPrinterLight[i], kLightLabel[i],
                    "Printer points (0.025 log E each). More light in a channel puts more of the complementary dye "
                    "in the print.",
                    0.0, -20.0, 20.0, -20.0, 20.0, 1.0, print);
    doubleParam(kParamPrintDensity, "Print density",
                "Printer points added to all three lights; more light = denser, darker print", 0.0, -20.0, 20.0, -20.0, 20.0,
                1.0, print);
}

OFX::ImageEffect* OpenFilmFactory::createInstance(OfxImageEffectHandle handle, OFX::ContextEnum /*context*/) {
    return new OpenFilmEffect(handle);
}

namespace OFX {
namespace Plugin {
// Called by the Support library's OfxGetPlugin(): the list of plugins in this binary.
void getPluginIDs(OFX::PluginFactoryArray& ids) {
    static OpenFilmFactory factory(kPluginIdentifier, 0, 1);  // version 0.1
    ids.push_back(&factory);
}
}  // namespace Plugin
}  // namespace OFX
