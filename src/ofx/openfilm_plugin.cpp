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
#include "ofxsImageEffect.h"

#include "core/cpu_renderer.hpp"
#include "metal/metal_renderer.h"

#include <dlfcn.h>

#include <memory>
#include <string>

namespace {

constexpr const char* kPluginIdentifier = "org.openfilm.OpenFilm";
constexpr const char* kPluginName = "OpenFilm";
constexpr const char* kParamNegativeGroup = "negative";
constexpr const char* kParamExposure = "exposure";

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
        openfilm::RenderParams params;
        params.exposureStops = exposure_->getValueAtTime(args.time);

        if (args.isEnabledMetalRender) {
            // GPU path: getPixelData() is an id<MTLBuffer>, not CPU memory.
            openfilm::MetalRenderer& gpu = metalRenderer();
            if (!gpu.isValid()) {
                setPersistentMessage(OFX::Message::eMessageError, "",
                                     "OpenFilm: GPU shaders failed to load: " + gpu.error());
                OFX::throwSuiteStatusException(kOfxStatFailed);
            }
            if (!gpu.render(args.pMetalCmdQ, src->getPixelData(), srcGeom, dst->getPixelData(), dstGeom, win, params))
                OFX::throwSuiteStatusException(kOfxStatFailed);
        } else {
            openfilm::renderCpu(openfilm::ConstImageView{static_cast<const float*>(src->getPixelData()), srcGeom},
                                openfilm::ImageView{static_cast<float*>(dst->getPixelData()), dstGeom}, win, params);
        }
    }

private:
    OFX::Clip* dstClip_ = nullptr;
    OFX::Clip* srcClip_ = nullptr;
    OFX::DoubleParam* exposure_ = nullptr;
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
    desc.setSupportsMetalRender(true);
}

void OpenFilmFactory::describeInContext(OFX::ImageEffectDescriptor& desc, OFX::ContextEnum /*context*/) {
    OFX::ClipDescriptor* src = desc.defineClip(kOfxImageEffectSimpleSourceClipName);
    src->addSupportedComponent(OFX::ePixelComponentRGBA);
    src->setSupportsTiles(false);

    OFX::ClipDescriptor* dst = desc.defineClip(kOfxImageEffectOutputClipName);
    dst->addSupportedComponent(OFX::ePixelComponentRGBA);
    dst->setSupportsTiles(false);

    OFX::GroupParamDescriptor* negative = desc.defineGroupParam(kParamNegativeGroup);
    negative->setLabel("Negative");
    negative->setOpen(true);

    OFX::DoubleParamDescriptor* exposure = desc.defineDoubleParam(kParamExposure);
    exposure->setLabel("Exposure");
    exposure->setHint("Negative exposure in stops");
    exposure->setDefault(0.0);
    exposure->setRange(-10.0, 10.0);
    exposure->setDisplayRange(-5.0, 5.0);
    exposure->setIncrement(0.1);
    exposure->setDoubleType(OFX::eDoubleTypePlain);
    exposure->setParent(*negative);
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
