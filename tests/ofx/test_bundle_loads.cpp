// Loads the built plugin the same way Resolve does (dlopen) and checks what Resolve checks first:
// the two exported OFX entry points and the plugin's identity. Also checks the bundle layout.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "core/film_model.hpp"
#include "metal/metal_renderer.h"
#include "ofxCore.h"

#include <dlfcn.h>

#include <fstream>
#include <sstream>
#include <string>

namespace {

const std::string kBundle = OPENFILM_BUNDLE_DIR;  // .../OpenFilm.ofx.bundle

std::string readFile(const std::string& path) {
    std::ifstream f(path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

bool exists(const std::string& path) { return std::ifstream(path).good(); }

}  // namespace

TEST_CASE("bundle exports OFX entry points") {
    void* lib = dlopen((kBundle + "/Contents/MacOS/OpenFilm.ofx").c_str(), RTLD_NOW | RTLD_LOCAL);
    CAPTURE(dlerror());
    REQUIRE(lib != nullptr);

    using CountFn = int (*)();
    using GetFn = OfxPlugin* (*)(int);
    auto count = reinterpret_cast<CountFn>(dlsym(lib, "OfxGetNumberOfPlugins"));
    auto get = reinterpret_cast<GetFn>(dlsym(lib, "OfxGetPlugin"));
    REQUIRE(count != nullptr);
    REQUIRE(get != nullptr);

    REQUIRE(count() == 1);
    OfxPlugin* plugin = get(0);
    REQUIRE(plugin != nullptr);
    CHECK(std::string(plugin->pluginIdentifier) == "org.openfilm.OpenFilm");
    CHECK(std::string(plugin->pluginApi) == "OfxImageEffectPluginAPI");
}

TEST_CASE("bundle ships metallib") {
    const std::string path = kBundle + "/Contents/Resources/OpenFilm.metallib";
    REQUIRE(exists(path));
    openfilm::MetalRenderer r(path);
    CAPTURE(r.error());
    CHECK(r.isValid());
}

TEST_CASE("Info.plist present") {
    const std::string path = kBundle + "/Contents/Info.plist";
    REQUIRE(exists(path));
    CHECK(readFile(path).find("OpenFilm.ofx") != std::string::npos);
}

TEST_CASE("bundle ships film resources") {
    const std::string res = kBundle + "/Contents/Resources";
    CHECK(exists(res + "/data/stocks/kodak-5219-type.json"));
    CHECK(exists(res + "/data/stocks/kodak-2383-type.json"));
    CHECK(exists(res + "/data/spectral/iso5_3_status_m.csv"));
    CHECK(exists(res + "/rec2020.coeff"));
}

TEST_CASE("film resources load from the bundle") {
    const auto r = openfilm::FilmResources::load(kBundle + "/Contents/Resources");
    REQUIRE(r != nullptr);
    CHECK(r->stocks.size() == 2);
}
