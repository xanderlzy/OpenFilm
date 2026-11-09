// Loads the built plugin the same way Resolve does (dlopen) and checks what Resolve checks first:
// the two exported OFX entry points and the plugin's identity. Also checks the bundle layout.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "core/film_model.hpp"
#include "metal/metal_renderer.h"
#include "ofxCore.h"
#include "ofxImageEffect.h"
#include "ofxMemory.h"
#include "ofxMessage.h"
#include "ofxMultiThread.h"
#include "ofxParam.h"
#include "ofxProperty.h"

#include <dlfcn.h>

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
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

// ---- A minimal recording host: enough of the OFX suites for the plugin to describe itself. ----
// Every property set is a name -> list of values store; every defined parameter is kept with its
// type and properties so the test can read back what the plugin declared.
namespace stub {

struct Val {
  std::string s;
  double d = 0;
  int i = 0;
  void* p = nullptr;
};
struct PropSet {
  std::map<std::string, std::vector<Val>> v;
};
struct Param {
  std::string type, name;
  PropSet props;
};
struct ParamSet {
  std::vector<std::unique_ptr<Param>> list;
  Param* find(const std::string& n) const {
    for (const auto& p : list) if (p->name == n) return p.get();
    return nullptr;
  }
};
struct Effect {
  PropSet props;
  ParamSet params;
  std::vector<std::unique_ptr<PropSet>> clips;
};

Val& slot(OfxPropertySetHandle h, const char* name, int index) {
  auto& vec = reinterpret_cast<PropSet*>(h)->v[name];
  if (int(vec.size()) <= index) vec.resize(size_t(index) + 1);
  return vec[size_t(index)];
}
template <class T> OfxStatus get(OfxPropertySetHandle h, const char* name, int index, T* out, T Val::*field) {
  const auto& m = reinterpret_cast<PropSet*>(h)->v;
  const auto it = m.find(name);
  if (it == m.end() || index >= int(it->second.size())) { *out = T(); return kOfxStatOK; }  // unknown: 0
  *out = it->second[size_t(index)].*field;
  return kOfxStatOK;
}

OfxStatus setPointer(OfxPropertySetHandle h, const char* n, int i, void* v) { slot(h, n, i).p = v; return kOfxStatOK; }
OfxStatus setString(OfxPropertySetHandle h, const char* n, int i, const char* v) { slot(h, n, i).s = v; return kOfxStatOK; }
OfxStatus setDouble(OfxPropertySetHandle h, const char* n, int i, double v) { slot(h, n, i).d = v; return kOfxStatOK; }
OfxStatus setInt(OfxPropertySetHandle h, const char* n, int i, int v) { slot(h, n, i).i = v; return kOfxStatOK; }
OfxStatus setPointerN(OfxPropertySetHandle h, const char* n, int c, void* const* v) { for (int k = 0; k < c; ++k) slot(h, n, k).p = v[k]; return kOfxStatOK; }
OfxStatus setStringN(OfxPropertySetHandle h, const char* n, int c, const char* const* v) { for (int k = 0; k < c; ++k) slot(h, n, k).s = v[k]; return kOfxStatOK; }
OfxStatus setDoubleN(OfxPropertySetHandle h, const char* n, int c, const double* v) { for (int k = 0; k < c; ++k) slot(h, n, k).d = v[k]; return kOfxStatOK; }
OfxStatus setIntN(OfxPropertySetHandle h, const char* n, int c, const int* v) { for (int k = 0; k < c; ++k) slot(h, n, k).i = v[k]; return kOfxStatOK; }
OfxStatus getPointer(OfxPropertySetHandle h, const char* n, int i, void** o) { return get(h, n, i, o, &Val::p); }
OfxStatus getString(OfxPropertySetHandle h, const char* n, int i, char** o) {
  const auto& m = reinterpret_cast<PropSet*>(h)->v;
  const auto it = m.find(n);
  static char empty[] = "";
  if (it == m.end() || i >= int(it->second.size())) { *o = empty; return kOfxStatOK; }
  *o = const_cast<char*>(it->second[size_t(i)].s.c_str());
  return kOfxStatOK;
}
OfxStatus getDouble(OfxPropertySetHandle h, const char* n, int i, double* o) { return get(h, n, i, o, &Val::d); }
OfxStatus getInt(OfxPropertySetHandle h, const char* n, int i, int* o) { return get(h, n, i, o, &Val::i); }
OfxStatus getDim(OfxPropertySetHandle h, const char* n, int* c) {
  const auto& m = reinterpret_cast<PropSet*>(h)->v;
  const auto it = m.find(n);
  *c = it == m.end() ? 0 : int(it->second.size());
  return kOfxStatOK;
}
OfxStatus propReset(OfxPropertySetHandle h, const char* n) { reinterpret_cast<PropSet*>(h)->v.erase(n); return kOfxStatOK; }

OfxStatus effGetPropertySet(OfxImageEffectHandle e, OfxPropertySetHandle* p) { *p = reinterpret_cast<OfxPropertySetHandle>(&reinterpret_cast<Effect*>(e)->props); return kOfxStatOK; }
OfxStatus effGetParamSet(OfxImageEffectHandle e, OfxParamSetHandle* p) { *p = reinterpret_cast<OfxParamSetHandle>(&reinterpret_cast<Effect*>(e)->params); return kOfxStatOK; }
OfxStatus clipDefine(OfxImageEffectHandle e, const char*, OfxPropertySetHandle* p) {
  auto* eff = reinterpret_cast<Effect*>(e);
  eff->clips.push_back(std::make_unique<PropSet>());
  *p = reinterpret_cast<OfxPropertySetHandle>(eff->clips.back().get());
  return kOfxStatOK;
}
OfxStatus paramDefine(OfxParamSetHandle ps, const char* type, const char* name, OfxPropertySetHandle* props) {
  auto* set = reinterpret_cast<ParamSet*>(ps);
  set->list.push_back(std::make_unique<Param>());
  set->list.back()->type = type;
  set->list.back()->name = name;
  if (props) *props = reinterpret_cast<OfxPropertySetHandle>(&set->list.back()->props);
  return kOfxStatOK;
}
OfxStatus paramSetProps(OfxParamSetHandle ps, OfxPropertySetHandle* p) {
  static PropSet dummy;
  (void)ps;
  *p = reinterpret_cast<OfxPropertySetHandle>(&dummy);
  return kOfxStatOK;
}
OfxStatus memAllocS(void*, size_t n, void** out) { *out = std::malloc(n); return kOfxStatOK; }
OfxStatus memFreeS(void* p) { std::free(p); return kOfxStatOK; }

const void* fetchSuite(OfxPropertySetHandle, const char* name, int version) {
  static OfxPropertySuiteV1 prop = {setPointer, setString, setDouble, setInt, setPointerN, setStringN, setDoubleN, setIntN,
                                    getPointer, getString, getDouble, getInt, nullptr, nullptr, nullptr, nullptr, propReset, getDim};
  static OfxImageEffectSuiteV1 eff = {};
  static OfxParameterSuiteV1 par = {};
  static OfxMemorySuiteV1 mem = {memAllocS, memFreeS};
  static OfxMultiThreadSuiteV1 mt = {};
  static OfxMessageSuiteV1 msg = {};
  static bool init = false;
  if (!init) {
    // The N getters share the single-value ones in this stub (the plugin never reads arrays).
    prop.propGetPointerN = nullptr; prop.propGetStringN = nullptr; prop.propGetDoubleN = nullptr; prop.propGetIntN = nullptr;
    eff.getPropertySet = effGetPropertySet;
    eff.getParamSet = effGetParamSet;
    eff.clipDefine = clipDefine;
    par.paramDefine = paramDefine;
    par.paramSetGetPropertySet = paramSetProps;
    init = true;
  }
  const std::string n = name;
  if (n == kOfxPropertySuite && version == 1) return &prop;
  if (n == kOfxImageEffectSuite && version == 1) return &eff;
  if (n == kOfxParameterSuite && version == 1) return &par;
  if (n == kOfxMemorySuite && version == 1) return &mem;
  if (n == kOfxMultiThreadSuite && version == 1) return &mt;
  if (n == kOfxMessageSuite && version == 1) return &msg;
  return nullptr;
}

double dbl(const Param* p, const char* k, int i = 0) { return p->props.v.at(k).at(size_t(i)).d; }
int integer(const Param* p, const char* k, int i = 0) { return p->props.v.at(k).at(size_t(i)).i; }
std::string str(const Param* p, const char* k, int i = 0) {
  const auto it = p->props.v.find(k);
  return it == p->props.v.end() || i >= int(it->second.size()) ? std::string() : it->second[size_t(i)].s;
}

}  // namespace stub

TEST_CASE("plugin describes the M4 controls") {
  void* lib = dlopen((kBundle + "/Contents/MacOS/OpenFilm.ofx").c_str(), RTLD_NOW | RTLD_LOCAL);
  REQUIRE(lib != nullptr);
  using GetFn = OfxPlugin* (*)(int);
  OfxPlugin* plugin = reinterpret_cast<GetFn>(dlsym(lib, "OfxGetPlugin"))(0);
  REQUIRE(plugin != nullptr);

  static OfxHost host;
  static stub::PropSet hostProps;
  host.host = reinterpret_cast<OfxPropertySetHandle>(&hostProps);
  host.fetchSuite = stub::fetchSuite;
  plugin->setHost(&host);

  REQUIRE(plugin->mainEntry(kOfxActionLoad, nullptr, nullptr, nullptr) == kOfxStatOK);
  stub::Effect effect;
  const auto h = reinterpret_cast<OfxImageEffectHandle>(&effect);
  REQUIRE(plugin->mainEntry(kOfxActionDescribe, h, nullptr, nullptr) == kOfxStatOK);
  // Resolve is offered the Metal path: still advertised.
  CHECK(stub::slot(reinterpret_cast<OfxPropertySetHandle>(&effect.props), "OfxImageEffectPropMetalRenderSupported", 0).s == "true");

  stub::PropSet in;
  stub::slot(reinterpret_cast<OfxPropertySetHandle>(&in), kOfxImageEffectPropContext, 0).s = kOfxImageEffectContextFilter;
  REQUIRE(plugin->mainEntry(kOfxImageEffectActionDescribeInContext, h, reinterpret_cast<OfxPropertySetHandle>(&in), nullptr) == kOfxStatOK);

  const auto param = [&](const char* name) {
    const stub::Param* p = effect.params.find(name);
    INFO("parameter " << name);
    REQUIRE(p != nullptr);
    return p;
  };
  const auto checkDouble = [&](const char* name, double def, double lo, double hi, const char* group) {
    const stub::Param* p = param(name);
    CHECK(p->type == kOfxParamTypeDouble);
    CHECK(stub::dbl(p, kOfxParamPropDefault) == doctest::Approx(def));
    CHECK(stub::dbl(p, kOfxParamPropMin) == doctest::Approx(lo));
    CHECK(stub::dbl(p, kOfxParamPropMax) == doctest::Approx(hi));
    CHECK(stub::str(p, kOfxParamPropParent) == group);
  };
  const auto checkBool = [&](const char* name, bool def, const char* label, const char* group) {
    const stub::Param* p = param(name);
    CHECK(p->type == kOfxParamTypeBoolean);
    CHECK((stub::integer(p, kOfxParamPropDefault) != 0) == def);
    CHECK(stub::str(p, kOfxPropLabel) == label);
    CHECK(stub::str(p, kOfxParamPropParent) == group);
  };

  for (const auto& g : {std::pair<const char*, const char*>{"formatGroup", "Film format"}, {"sharpnessGroup", "Sharpness"},
                        {"halationGroup", "Halation"}, {"bloomGroup", "Bloom"}}) {
    CHECK(param(g.first)->type == kOfxParamTypeGroup);
    CHECK(stub::str(param(g.first), kOfxPropLabel) == g.second);
  }
  const stub::Param* fmt = param("filmFormat");
  CHECK(fmt->type == kOfxParamTypeChoice);
  CHECK(stub::str(fmt, kOfxPropLabel) == "Format");
  CHECK(fmt->props.v.at(kOfxParamPropChoiceOption).size() == 6);
  CHECK(stub::str(fmt, kOfxParamPropChoiceOption, 0).find("Super 35") == 0);
  CHECK(stub::integer(fmt, kOfxParamPropDefault) == 0);
  CHECK(stub::str(fmt, kOfxParamPropParent) == "formatGroup");
  checkDouble("gateWidth", 24.89, 5, 80, "formatGroup");
  CHECK(stub::str(param("gateWidth"), kOfxPropLabel) == "Gate width (mm)");
  checkBool("sharpnessEnable", true, "Enable", "sharpnessGroup");
  checkBool("halationEnable", true, "Enable", "halationGroup");
  checkBool("halationRemjet", true, "Rem-jet", "halationGroup");
  checkDouble("halationStrength", 1, 0, 5, "halationGroup");
  CHECK(stub::str(param("halationStrength"), kOfxPropLabel) == "Strength");
  checkDouble("halationRadius", 1, 0.25, 4, "halationGroup");
  CHECK(stub::str(param("halationRadius"), kOfxPropLabel) == "Radius");
  checkBool("bloomEnable", false, "Enable", "bloomGroup");
  checkDouble("bloomAmount", 0.25, 0, 1, "bloomGroup");
  CHECK(stub::str(param("bloomAmount"), kOfxPropLabel) == "Amount");
  checkDouble("bloomRadius", 0.5, 0.05, 5, "bloomGroup");
  CHECK(stub::str(param("bloomRadius"), kOfxPropLabel) == "Radius (mm)");
  // The M3 controls are still there.
  for (const char* n : {"exposure", "negativeStock", "printStock", "printerLightR", "printDensity"}) CHECK(effect.params.find(n) != nullptr);
}
