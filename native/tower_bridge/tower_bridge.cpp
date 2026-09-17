// Read-only ARM64 IL2CPP bridge for the private instrumented-training profile.
// It contains no game offsets, assets, or action invocation path.

#include <android/log.h>
#include <arpa/inet.h>
#include <dlfcn.h>
#include <netinet/in.h>
#include <pthread.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cmath>
#include <cstddef>
#include <ctime>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifndef TOWER_BRIDGE_PACKAGE_VERSION
#define TOWER_BRIDGE_PACKAGE_VERSION "unconfigured"
#endif
#ifndef TOWER_BRIDGE_PACKAGE_VERSION_CODE
#define TOWER_BRIDGE_PACKAGE_VERSION_CODE 0
#endif
#ifndef TOWER_BRIDGE_OFFICIAL_SIGNER_SHA256
#define TOWER_BRIDGE_OFFICIAL_SIGNER_SHA256 "unconfigured"
#endif
#ifndef TOWER_BRIDGE_ORIGINAL_LIBUNITY_SHA256
#define TOWER_BRIDGE_ORIGINAL_LIBUNITY_SHA256 "unconfigured"
#endif
#ifndef TOWER_BRIDGE_LIBIL2CPP_SHA256
#define TOWER_BRIDGE_LIBIL2CPP_SHA256 "unconfigured"
#endif
#ifndef TOWER_BRIDGE_UNITY_VERSION
#define TOWER_BRIDGE_UNITY_VERSION "6000.3.15f1"
#endif
#ifndef TOWER_BRIDGE_METADATA_VERSION
#define TOWER_BRIDGE_METADATA_VERSION 39
#endif
#ifndef TOWER_BRIDGE_VERSION
#define TOWER_BRIDGE_VERSION "tower-bridge-v1"
#endif
#ifndef TOWER_BRIDGE_PROFILE_ID
#define TOWER_BRIDGE_PROFILE_ID "unconfigured"
#endif
#ifndef TOWER_BRIDGE_MAIN_GAME_OBJECT
#define TOWER_BRIDGE_MAIN_GAME_OBJECT "Main"
#endif

namespace {

constexpr char kLogTag[] = "tower_bridge";
constexpr uint16_t kLoopbackPort = 47651;
constexpr uint32_t kMaxFrameBytes = 65536;
constexpr size_t kMaxEntriesPerFamily = 64;
constexpr useconds_t kObservationIntervalMicros = 250000;
constexpr useconds_t kHeartbeatIntervalMicros = 1000000;
constexpr useconds_t kIl2CppInitializationDelayMicros = 4000000;
constexpr useconds_t kCommandTimeoutMicros = 3000000;
constexpr useconds_t kCommandPollMicros = 50000;
constexpr useconds_t kLifecycleTimeoutMicros = 30000000;
constexpr useconds_t kLifecyclePollMicros = 250000;
// Floors the cadence so a very high speed cannot spin the loop, but it must stay
// well below the game-time interval or it becomes the binding constraint and
// starves the policy of decisions per game second: at 20 ms it bound above 12.5x
// and cut decisions per episode from 528 to 162 between 1.5x and 32x.
constexpr useconds_t kMinIntervalMicros = 4000;
// How often to look for the next rendered frame. Well under one frame at any
// plausible rate, so an advance notices each frame rather than overshooting it.
constexpr useconds_t kFramePollMicros = 2000;
// A hard wall-clock ceiling on one advance, measured against CLOCK_MONOTONIC so
// it is a real ceiling. Game time is bounded by the requested budget, but a
// stalled renderer would otherwise never end the loop. The host's read timeout
// must cover this plus the settle below: `DEFAULT_READ_TIMEOUT_SECONDS` in
// `src/tower_rl/infrastructure/instrumented_bridge.py` is derived from these two.
constexpr useconds_t kAdvanceWallBudgetMicros = 15000000;
// `Pause` is dispatched to Unity's main thread and lands a frame or two later.
// `MainFields` resolves no game-owned pause flag, so the landing is observed
// through the frames the game still renders: the advance keeps counting them at
// the same game-time weight until two more have gone by, or this much wall time
// has, and only then reads the state it reports.
constexpr useconds_t kPauseSettleMicros = 500000;
constexpr int32_t kPauseSettleFrames = 2;
// The width Python maps into its action mask, per upgrade family. A slot the
// game does not report is unavailable here exactly as it is masked there; it is
// `SLOTS_PER_FAMILY` in `src/tower_rl/domain/run_actions.py` and the two must
// agree or the bridge would stop on availability the host cannot act on.
constexpr size_t kMaskSlotsPerFamily = 20;

struct Il2CppDomain;
struct Il2CppThread;
struct Il2CppAssembly;
struct Il2CppImage;
struct Il2CppClass;
struct FieldInfo;
struct Il2CppObject;
struct Il2CppArray;

struct Il2CppApi {
  Il2CppDomain* (*domain_get)() = nullptr;
  Il2CppThread* (*thread_attach)(Il2CppDomain*) = nullptr;
  void (*thread_detach)(Il2CppThread*) = nullptr;
  const Il2CppAssembly** (*domain_get_assemblies)(Il2CppDomain*, size_t*) = nullptr;
  const Il2CppImage* (*assembly_get_image)(const Il2CppAssembly*) = nullptr;
  size_t (*image_get_class_count)(const Il2CppImage*) = nullptr;
  Il2CppClass* (*image_get_class)(const Il2CppImage*, size_t) = nullptr;
  const char* (*class_get_name)(Il2CppClass*) = nullptr;
  const char* (*class_get_namespace)(Il2CppClass*) = nullptr;
  FieldInfo* (*class_get_field_from_name)(Il2CppClass*, const char*) = nullptr;
  void (*field_get_value)(Il2CppObject*, FieldInfo*, void*) = nullptr;
  void (*field_static_get_value)(FieldInfo*, void*) = nullptr;
  void (*field_static_set_value)(FieldInfo*, void*) = nullptr;
  Il2CppClass* (*object_get_class)(Il2CppObject*) = nullptr;
  size_t (*array_length)(Il2CppArray*) = nullptr;
  size_t (*array_get_byte_length)(Il2CppArray*) = nullptr;
  size_t (*array_object_header_size)() = nullptr;
  size_t (*array_element_size)(Il2CppClass*) = nullptr;
};

template <typename T>
bool Resolve(void* handle, const char* name, T* target) {
  *target = reinterpret_cast<T>(dlsym(handle, name));
  return *target != nullptr;
}

bool ResolveApi(Il2CppApi* api) {
  void* handle = dlopen("libil2cpp.so", RTLD_NOW | RTLD_NOLOAD);
  if (handle == nullptr) return false;
  return Resolve(handle, "il2cpp_domain_get", &api->domain_get) &&
         Resolve(handle, "il2cpp_thread_attach", &api->thread_attach) &&
         Resolve(handle, "il2cpp_thread_detach", &api->thread_detach) &&
         Resolve(handle, "il2cpp_domain_get_assemblies", &api->domain_get_assemblies) &&
         Resolve(handle, "il2cpp_assembly_get_image", &api->assembly_get_image) &&
         Resolve(handle, "il2cpp_image_get_class_count", &api->image_get_class_count) &&
         Resolve(handle, "il2cpp_image_get_class", &api->image_get_class) &&
         Resolve(handle, "il2cpp_class_get_name", &api->class_get_name) &&
         Resolve(handle, "il2cpp_class_get_namespace", &api->class_get_namespace) &&
         Resolve(handle, "il2cpp_class_get_field_from_name", &api->class_get_field_from_name) &&
         Resolve(handle, "il2cpp_field_get_value", &api->field_get_value) &&
         Resolve(handle, "il2cpp_field_static_get_value", &api->field_static_get_value) &&
         Resolve(handle, "il2cpp_field_static_set_value", &api->field_static_set_value) &&
         Resolve(handle, "il2cpp_object_get_class", &api->object_get_class) &&
         Resolve(handle, "il2cpp_array_length", &api->array_length) &&
         Resolve(handle, "il2cpp_array_get_byte_length", &api->array_get_byte_length) &&
         Resolve(handle, "il2cpp_array_object_header_size", &api->array_object_header_size) &&
         Resolve(handle, "il2cpp_array_element_size", &api->array_element_size);
}

using UnitySendMessage = void (*)(const char*, const char*, const char*);

UnitySendMessage ResolveUnitySendMessage() {
  void* unity = dlopen("libunity.so", RTLD_NOW | RTLD_NOLOAD);
  return unity == nullptr ? nullptr : reinterpret_cast<UnitySendMessage>(dlsym(unity, "UnitySendMessage"));
}

Il2CppClass* FindMainClass(const Il2CppApi& api, Il2CppDomain* domain) {
  size_t assemblies_count = 0;
  const Il2CppAssembly** assemblies = api.domain_get_assemblies(domain, &assemblies_count);
  for (size_t assembly_index = 0; assembly_index < assemblies_count; ++assembly_index) {
    const Il2CppImage* image = api.assembly_get_image(assemblies[assembly_index]);
    for (size_t class_index = 0; class_index < api.image_get_class_count(image); ++class_index) {
      Il2CppClass* candidate = api.image_get_class(image, class_index);
      const char* name = candidate == nullptr ? nullptr : api.class_get_name(candidate);
      const char* name_space = candidate == nullptr ? nullptr : api.class_get_namespace(candidate);
      if (name != nullptr && name_space != nullptr && std::strcmp(name, "Main") == 0 &&
          std::strcmp(name_space, "") == 0) {
        return candidate;
      }
    }
  }
  return nullptr;
}

#ifdef TOWER_BRIDGE_DIAGNOSTICS
// Private, build-flag-gated inventory used to locate semantic members on a new
// game build. It reads names through exported IL2CPP APIs only, never a metadata
// dump, and is absent from an ordinary build.
struct MethodInfo;

void LogClassMembers(Il2CppClass* klass, const char* label) {
  void* il2cpp = dlopen("libil2cpp.so", RTLD_NOW | RTLD_NOLOAD);
  if (il2cpp == nullptr || klass == nullptr) return;
  const MethodInfo* (*class_get_methods)(Il2CppClass*, void**) = nullptr;
  const char* (*method_get_name)(const MethodInfo*) = nullptr;
  uint32_t (*method_get_param_count)(const MethodInfo*) = nullptr;
  FieldInfo* (*class_get_fields)(Il2CppClass*, void**) = nullptr;
  const char* (*field_get_name)(FieldInfo*) = nullptr;
  if (!Resolve(il2cpp, "il2cpp_class_get_methods", &class_get_methods) ||
      !Resolve(il2cpp, "il2cpp_method_get_name", &method_get_name) ||
      !Resolve(il2cpp, "il2cpp_method_get_param_count", &method_get_param_count)) {
    return;
  }
  void* iterator = nullptr;
  for (const MethodInfo* method = class_get_methods(klass, &iterator); method != nullptr;
       method = class_get_methods(klass, &iterator)) {
    const char* name = method_get_name(method);
    if (name != nullptr) {
      __android_log_print(ANDROID_LOG_INFO, kLogTag, "%s.method %s/%u", label, name,
                          method_get_param_count(method));
    }
  }
  if (!Resolve(il2cpp, "il2cpp_class_get_fields", &class_get_fields) ||
      !Resolve(il2cpp, "il2cpp_field_get_name", &field_get_name)) {
    return;
  }
  iterator = nullptr;
  for (FieldInfo* field = class_get_fields(klass, &iterator); field != nullptr;
       field = class_get_fields(klass, &iterator)) {
    const char* name = field_get_name(field);
    if (name != nullptr) {
      __android_log_print(ANDROID_LOG_INFO, kLogTag, "%s.field %s", label, name);
    }
  }
}
#endif

Il2CppClass* FindClass(const Il2CppApi& api, Il2CppDomain* domain, const char* wanted) {
  size_t assemblies_count = 0;
  const Il2CppAssembly** assemblies = api.domain_get_assemblies(domain, &assemblies_count);
  for (size_t assembly = 0; assembly < assemblies_count; ++assembly) {
    const Il2CppImage* image = api.assembly_get_image(assemblies[assembly]);
    for (size_t index = 0; index < api.image_get_class_count(image); ++index) {
      Il2CppClass* candidate = api.image_get_class(image, index);
      const char* name = candidate == nullptr ? nullptr : api.class_get_name(candidate);
      if (name != nullptr && std::strcmp(name, wanted) == 0) return candidate;
    }
  }
  return nullptr;
}

struct FamilyFields {
  const char* name;
  FieldInfo* cost;
  FieldInfo* level;
  FieldInfo* unlocked;
  FieldInfo* tier_unlocked;
  FieldInfo* max_level;
  FieldInfo* maxed;
};

struct MainFields {
  FieldInfo* instance;
  FieldInfo* game_speed;
  FieldInfo* game_max_speed;
  FieldInfo* play_time;
  // The game's own per-round clock, and the only witness that `captureDeltaTime`
  // really made a frame worth what the advance asked for. `playTime` cannot be:
  // it is the account-lifetime clock and advances at wall rate whatever the
  // game clock does. Stored as a `float` by the game, unlike `playTime`.
  FieldInfo* round_time;
  FieldInfo* cash;
  FieldInfo* current_wave;
  FieldInfo* tower_health;
  FieldInfo* tower_max_health;
  FieldInfo* game_over;
  FieldInfo* round_active;
  FieldInfo* upgrade_select;
  FamilyFields attack;
  FamilyFields defense;
  FamilyFields utility;
};

struct Runtime {
  Il2CppApi api;
  MainFields fields;
  UnitySendMessage unity_send_message = nullptr;
};

struct UpgradeEvidence {
  double cash;
  double cost;
  int32_t level;
  uint8_t unlocked;
  uint8_t tier_unlocked;
  uint8_t maxed;
};

// The game's own parameterless entry points. Navigation is controller-owned and
// never a policy action, so lifecycle commands are a separate kind from the
// `advance` and `buy_upgrade` a policy decision turns into.
struct LifecycleAction {
  const char* name;
  const char* method;
  bool expect_active;
};

constexpr LifecycleAction kLifecycleActions[] = {
    {"start_round", "StartNewRoundFunction", true},
    {"retry", "AutoRetryBattle", true},
    {"go_home", "Button_GameEndPanelGoHome", false},
    // The game's own auto-restart toggle. Dispatched from a terminal run it is
    // self-confirming: either the game starts the next round by itself, or the
    // wait expires and the actor is quarantined rather than assumed healthy.
    {"enable_auto_restart", "Button_ToggleAutoRestartBattle", true},
    // Speed is the game's own control. The observation reports `game_speed`, so
    // the host verifies the effect instead of assuming it.
    {"speed_max", "SpeedChangeMax", true},
    {"speed_down", "SpeedChangeDown", true},
    // Pause makes the environment turn-based: thinking then costs no game time.
    {"pause", "Pause", true},
    {"unpause", "Unpause", true},
};

// Each family recomputes its own costs. The game only refreshes a family while
// its tab is displayed, so an actor that never touches the screen must ask for
// the recalculation itself.
constexpr const char* kCostRefreshMethods[] = {
    "UpgradeCostCalc", "UpgradeDefenseCostCalc", "UpgradeUtilityCostCalc"};

constexpr uint32_t kMinAdvanceBudgetMillis = 10;
constexpr uint32_t kMaxAdvanceBudgetMillis = 10000;
constexpr double kMinFrameGameMillis = 1.0;
constexpr double kMaxFrameGameMillis = 250.0;
// The protocol bound is deliberately wider than any endorsed speed. Unity clamps
// how much game time one frame may advance, so the usable ceiling is set by the
// achieved frame rate rather than by this number, and which speeds are actually
// admissible is decided by the equivalence gate, not by the protocol.
constexpr float kMinRequestedSpeed = 0.5F;
constexpr float kMaxRequestedSpeed = 64.0F;

struct Command {
  char request_id[65];
  uint64_t expected_sequence;
  const char* family;
  size_t index;
  const LifecycleAction* lifecycle;
  bool set_speed;
  float speed;
  bool advance;
  uint32_t budget_game_millis;
  float frame_game_millis;
  float health_change_fraction;
};

FieldInfo* Field(const Il2CppApi& api, Il2CppClass* klass, const char* name) {
  return api.class_get_field_from_name(klass, name);
}

bool LoadFields(const Il2CppApi& api, Il2CppClass* main, Il2CppClass* int_select, MainFields* fields) {
  fields->instance = Field(api, main, "<Instance>k__BackingField");
  fields->game_speed = Field(api, main, "gameSpeed");
  fields->game_max_speed = Field(api, main, "gameMaxSpeed");
  fields->play_time = Field(api, main, "playTime");
  // `roundTime` tracks this field identically on the pinned build (M1B-E017);
  // the gameplay clock is the one named for what it measures.
  fields->round_time = Field(api, main, "gameplayTimeThisRound");
  fields->cash = Field(api, main, "cash");
  fields->current_wave = Field(api, main, "currentWave");
  fields->tower_health = Field(api, main, "towerHealth");
  fields->tower_max_health = Field(api, main, "towerMaxHealth");
  fields->game_over = Field(api, main, "gameOverBool");
  fields->round_active = Field(api, main, "roundActiveBool");
  fields->upgrade_select = Field(api, int_select, "upgradeSelect");
  fields->attack = {"attack", Field(api, main, "upgradeCost"), Field(api, main, "upgradeLevel"),
                    Field(api, main, "upgradeUnlocked"), Field(api, main, "upgradeTierUnlocked"),
                    Field(api, main, "upgradeMaxLevel"), Field(api, main, "upgradesMaxedBool")};
  fields->defense = {"defense", Field(api, main, "upgradeDefenseCost"),
                     Field(api, main, "upgradeDefenseLevel"),
                     Field(api, main, "upgradeDefenseUnlocked"),
                     Field(api, main, "upgradeDefenseTierUnlocked"),
                     Field(api, main, "upgradeDefenseMaxLevel"),
                     Field(api, main, "upgradesDefenseMaxedBool")};
  fields->utility = {"utility", Field(api, main, "upgradeUtilityCost"),
                     Field(api, main, "upgradeUtilityLevel"),
                     Field(api, main, "upgradeUtilityUnlocked"),
                     Field(api, main, "upgradeUtilityTierUnlocked"),
                     Field(api, main, "upgradeUtilityMaxLevel"),
                     Field(api, main, "upgradesUtilityMaxedBool")};
  const FamilyFields families[] = {fields->attack, fields->defense, fields->utility};
  if (fields->instance == nullptr || fields->game_speed == nullptr || fields->cash == nullptr ||
      fields->current_wave == nullptr || fields->tower_health == nullptr ||
      fields->tower_max_health == nullptr || fields->game_over == nullptr ||
      fields->round_active == nullptr || fields->round_time == nullptr ||
      fields->upgrade_select == nullptr) return false;
  for (const FamilyFields& family : families) {
    if (family.cost == nullptr || family.level == nullptr || family.unlocked == nullptr ||
        family.tier_unlocked == nullptr || family.max_level == nullptr || family.maxed == nullptr) {
      return false;
    }
  }
  return true;
}

template <typename T>
bool ReadField(const Il2CppApi& api, Il2CppObject* object, FieldInfo* field, T* value) {
  if (object == nullptr || field == nullptr) return false;
  api.field_get_value(object, field, value);
  return true;
}

// Unity keeps a destroyed object's managed wrapper alive with a null native
// handle - the "fake null" that makes `obj == null` true in C# while the pointer
// is not null at all. Testing the managed pointer alone therefore cannot tell a
// live component from a destroyed one, and a finished run whose `Main` has been
// torn down would be read field by field and reported as a live observation.
bool NativeHandleIsAlive(const Il2CppApi& api, Il2CppObject* object) {
  if (object == nullptr) return false;
  Il2CppClass* klass = api.object_get_class(object);
  FieldInfo* cached =
      klass == nullptr ? nullptr : api.class_get_field_from_name(klass, "m_CachedPtr");
  // Not a UnityEngine.Object: the managed null test is the only one available and
  // is also sufficient, so absence of the field is not a failure.
  if (cached == nullptr) return true;
  void* handle = nullptr;
  api.field_get_value(object, cached, &handle);
  return handle != nullptr;
}

template <typename T>
bool ReadPrimitiveArray(const Il2CppApi& api, Il2CppArray* array, size_t index, T* value) {
  if (array == nullptr || index >= api.array_length(array)) return false;
  const size_t header_size = api.array_object_header_size();
  const size_t byte_length = api.array_get_byte_length(array);
  const size_t length = api.array_length(array);
  Il2CppClass* array_class = api.object_get_class(reinterpret_cast<Il2CppObject*>(array));
  if (array_class == nullptr) return false;
  const size_t element_size = api.array_element_size(array_class);
  if (header_size == 0 || length == 0 || element_size != sizeof(T) ||
      byte_length / length != element_size ||
      byte_length < (index + 1) * sizeof(T)) return false;
  const auto* bytes = reinterpret_cast<const uint8_t*>(array) + header_size + index * sizeof(T);
  std::memcpy(value, bytes, sizeof(T));
  return true;
}

const FamilyFields* Family(const MainFields& fields, const char* name) {
  if (std::strcmp(name, "attack") == 0) return &fields.attack;
  if (std::strcmp(name, "defense") == 0) return &fields.defense;
  if (std::strcmp(name, "utility") == 0) return &fields.utility;
  return nullptr;
}

bool ReadUpgradeEvidence(const Il2CppApi& api, const MainFields& fields, const char* family,
                         size_t index, UpgradeEvidence* evidence) {
  const FamilyFields* selected = Family(fields, family);
  Il2CppObject* main = nullptr;
  api.field_static_get_value(fields.instance, &main);
  if (selected == nullptr || !NativeHandleIsAlive(api, main)) return false;
  Il2CppArray *costs = nullptr, *levels = nullptr, *unlocked = nullptr, *tier = nullptr, *maxed = nullptr;
  if (!ReadField(api, main, fields.cash, &evidence->cash) ||
      !ReadField(api, main, selected->cost, &costs) || !ReadField(api, main, selected->level, &levels) ||
      !ReadField(api, main, selected->unlocked, &unlocked) ||
      !ReadField(api, main, selected->tier_unlocked, &tier) || !ReadField(api, main, selected->maxed, &maxed) ||
      costs == nullptr || levels == nullptr || unlocked == nullptr || tier == nullptr || maxed == nullptr ||
      index >= api.array_length(costs) || api.array_length(levels) != api.array_length(costs) ||
      api.array_length(unlocked) != api.array_length(costs) || api.array_length(tier) != api.array_length(costs) ||
      api.array_length(maxed) != api.array_length(costs)) return false;
  return ReadPrimitiveArray(api, costs, index, &evidence->cost) &&
         ReadPrimitiveArray(api, levels, index, &evidence->level) &&
         ReadPrimitiveArray(api, unlocked, index, &evidence->unlocked) &&
         ReadPrimitiveArray(api, tier, index, &evidence->tier_unlocked) &&
         ReadPrimitiveArray(api, maxed, index, &evidence->maxed) && std::isfinite(evidence->cash) &&
         std::isfinite(evidence->cost) && evidence->level >= 0 && evidence->unlocked <= 1 &&
         evidence->tier_unlocked <= 1 && evidence->maxed <= 1;
}

bool AppendFamily(const Il2CppApi& api, Il2CppObject* main, const FamilyFields& fields,
                  std::string* json) {
  Il2CppArray *costs = nullptr, *levels = nullptr, *unlocked = nullptr, *tier_unlocked = nullptr,
              *max_levels = nullptr, *maxed = nullptr;
  if (!ReadField(api, main, fields.cost, &costs) || !ReadField(api, main, fields.level, &levels) ||
      !ReadField(api, main, fields.unlocked, &unlocked) ||
      !ReadField(api, main, fields.tier_unlocked, &tier_unlocked) ||
      !ReadField(api, main, fields.max_level, &max_levels) ||
      !ReadField(api, main, fields.maxed, &maxed)) {
    __android_log_print(ANDROID_LOG_ERROR, kLogTag, "%s inventory fields unavailable", fields.name);
    return false;
  }
  if (costs == nullptr || levels == nullptr || unlocked == nullptr || tier_unlocked == nullptr ||
      max_levels == nullptr || maxed == nullptr) {
    __android_log_print(ANDROID_LOG_ERROR, kLogTag, "%s inventory contains a null array", fields.name);
    return false;
  }
  const size_t count = api.array_length(costs);
  if (count > kMaxEntriesPerFamily || api.array_length(levels) != count ||
      api.array_length(unlocked) != count || api.array_length(tier_unlocked) != count ||
      api.array_length(max_levels) != count || api.array_length(maxed) != count) {
    __android_log_print(ANDROID_LOG_ERROR, kLogTag,
                        "%s inventory length mismatch: cost=%zu level=%zu unlocked=%zu tier=%zu max=%zu maxed=%zu",
                        fields.name, count, api.array_length(levels), api.array_length(unlocked),
                        api.array_length(tier_unlocked), api.array_length(max_levels),
                        api.array_length(maxed));
    return false;
  }
  for (size_t index = 0; index < count; ++index) {
    double cost = 0.0;
    int32_t level = 0, max_level = 0;
    uint8_t is_unlocked = 0, is_tier_unlocked = 0, is_maxed = 0;
    if (!ReadPrimitiveArray(api, costs, index, &cost) ||
        !ReadPrimitiveArray(api, levels, index, &level) ||
        !ReadPrimitiveArray(api, max_levels, index, &max_level) ||
        !ReadPrimitiveArray(api, unlocked, index, &is_unlocked) ||
        !ReadPrimitiveArray(api, tier_unlocked, index, &is_tier_unlocked) ||
        !ReadPrimitiveArray(api, maxed, index, &is_maxed) ||
        !std::isfinite(cost) || level < 0 || max_level < 0 || is_unlocked > 1 ||
        is_tier_unlocked > 1 || is_maxed > 1) {
      __android_log_print(ANDROID_LOG_ERROR, kLogTag,
                          "%s inventory entry %zu has an invalid primitive value or layout", fields.name,
                          index);
      return false;
    }
    char entry[256];
    const int written = std::snprintf(entry, sizeof(entry),
                                      "%s{\"family\":\"%s\",\"index\":%zu,\"cost\":%.17g,"
                                      "\"level\":%d,\"max_level\":%d,\"unlocked\":%s,"
                                      "\"tier_unlocked\":%s,\"maxed\":%s}",
                                      json->back() == '[' ? "" : ",", fields.name, index, cost, level,
                                      max_level, is_unlocked ? "true" : "false",
                                      is_tier_unlocked ? "true" : "false",
                                      is_maxed ? "true" : "false");
    if (written < 0 || static_cast<size_t>(written) >= sizeof(entry)) return false;
    json->append(entry, static_cast<size_t>(written));
  }
  return true;
}

// A run that has not been initialized is an ordinary state, not a failure: the
// game sits at its home screen between episodes and the controller must still be
// able to start one. It is reported as its own message rather than an
// observation, so no plausible default is ever presented as run state.
enum class ObservationResult { kOk, kNoRun, kError };

ObservationResult BuildObservation(const Il2CppApi& api, const MainFields& fields,
                                   uint64_t sequence, std::string* json) {
  Il2CppObject* main = nullptr;
  api.field_static_get_value(fields.instance, &main);
  if (!NativeHandleIsAlive(api, main)) {
    return ObservationResult::kNoRun;
  }
  double cash = 0.0, health = 0.0, max_health = 0.0;
  int32_t wave = 0;
  uint8_t game_over = 0, round_active = 0;
  if (!ReadField(api, main, fields.cash, &cash) ||
      !ReadField(api, main, fields.current_wave, &wave) ||
      !ReadField(api, main, fields.tower_health, &health) ||
      !ReadField(api, main, fields.tower_max_health, &max_health) ||
      !ReadField(api, main, fields.game_over, &game_over) ||
      !ReadField(api, main, fields.round_active, &round_active) || !std::isfinite(cash) ||
      !std::isfinite(health) || !std::isfinite(max_health) || wave < 0 || game_over > 1 ||
      round_active > 1) {
    __android_log_print(ANDROID_LOG_ERROR, kLogTag,
                        "Main scalar observation is invalid: wave=%d cash=%g health=%g max_health=%g game_over=%u round_active=%u",
                        wave, cash, health, max_health, static_cast<unsigned>(game_over),
                        static_cast<unsigned>(round_active));
    return ObservationResult::kNoRun;
  }
  float game_speed = 0.0F;
  api.field_static_get_value(fields.game_speed, &game_speed);
  if (!std::isfinite(game_speed) || game_speed < 0.0F) return ObservationResult::kError;
  double play_time = 0.0;
  if (!ReadField(api, main, fields.play_time, &play_time) || !std::isfinite(play_time) ||
      play_time < 0.0) {
    return ObservationResult::kError;
  }
  const char* lifecycle = game_over ? "terminal" : (round_active ? "active" : "idle");
  char prefix[512];
  const int written = std::snprintf(prefix, sizeof(prefix),
                                    "{\"type\":\"observation\",\"sequence\":%llu,"
                                    "\"lifecycle\":\"%s\",\"wave\":%d,\"cash\":%.17g,"
                                    "\"health\":%.17g,\"max_health\":%.17g,\"terminal\":%s,"
                                    "\"round_active\":%s,\"game_speed\":%.9g,\"play_time\":%.17g,"
                                    "\"upgrades\":[",
                                    static_cast<unsigned long long>(sequence), lifecycle, wave, cash, health,
                                    max_health, game_over ? "true" : "false",
                                    round_active ? "true" : "false", game_speed, play_time);
  if (written < 0 || static_cast<size_t>(written) >= sizeof(prefix)) return ObservationResult::kError;
  *json = prefix;
  const bool complete = AppendFamily(api, main, fields.attack, json) &&
                        AppendFamily(api, main, fields.defense, json) &&
                        AppendFamily(api, main, fields.utility, json) &&
                        (json->append("]}"), json->size() <= kMaxFrameBytes);
  return complete ? ObservationResult::kOk : ObservationResult::kError;
}

bool WriteAll(int client, const void* data, size_t size) {
  const auto* bytes = static_cast<const uint8_t*>(data);
  while (size > 0) {
    const ssize_t written = send(client, bytes, size, MSG_NOSIGNAL);
    if (written < 0 && errno == EINTR) continue;
    if (written <= 0) return false;
    bytes += written;
    size -= static_cast<size_t>(written);
  }
  return true;
}

bool SendFrame(int client, const std::string& payload) {
  if (payload.empty() || payload.size() > kMaxFrameBytes) return false;
  uint32_t size = htonl(static_cast<uint32_t>(payload.size()));
  return WriteAll(client, &size, sizeof(size)) && WriteAll(client, payload.data(), payload.size());
}

bool SendError(int client, const char* code, const char* message) {
  return SendFrame(client, std::string("{\"type\":\"error\",\"code\":\"") + code +
                               "\",\"message\":\"" + message + "\"}");
}

bool ReadAll(int client, void* target, size_t size) {
  auto* bytes = static_cast<uint8_t*>(target);
  while (size > 0) { const ssize_t received = recv(client, bytes, size, 0); if (received <= 0) return false; bytes += received; size -= static_cast<size_t>(received); }
  return true;
}

bool ReadInboundFrame(int client, std::string* payload) {
  uint32_t network_size = 0;
  if (!ReadAll(client, &network_size, sizeof(network_size))) return false;
  const uint32_t size = ntohl(network_size);
  if (size == 0 || size > kMaxFrameBytes) return false;
  payload->assign(size, '\0');
  return ReadAll(client, payload->data(), size);
}

// Read one JSON number that must end exactly where the caller says it does.
bool ParseNumber(const std::string& text, double* value) {
  if (text.empty()) return false;
  char* end = nullptr;
  *value = std::strtod(text.c_str(), &end);
  return end != nullptr && *end == '\0' && std::isfinite(*value);
}

bool ParseCommand(const std::string& payload, Command* command) {
  constexpr char kPrefix[] = "{\"type\":\"command\",\"protocol_version\":1,\"request_id\":\"";
  constexpr char kSequenceKey[] = "\",\"expected_observation_sequence\":";
  if (payload.rfind(kPrefix, 0) != 0) return false;
  const size_t id_start = sizeof(kPrefix) - 1, id_end = payload.find(kSequenceKey, id_start);
  if (id_end == std::string::npos || id_end == id_start || id_end - id_start > 64) return false;
  std::memcpy(command->request_id, payload.data() + id_start, id_end - id_start); command->request_id[id_end - id_start] = '\0';
  for (size_t i = 0; command->request_id[i]; ++i) if (!std::strchr("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-", command->request_id[i])) return false;
  const size_t sequence_start = id_end + sizeof(kSequenceKey) - 1,
               kind = payload.find(",\"kind\":\"", sequence_start);
  if (kind == std::string::npos) return false;
  command->expected_sequence = 0;
  for (size_t i = sequence_start; i < kind; ++i) { if (payload[i] < '0' || payload[i] > '9') return false; command->expected_sequence = command->expected_sequence * 10 + (payload[i] - '0'); }
  if (command->expected_sequence == 0) return false;
  const std::string tail = payload.substr(kind + 9);
  command->lifecycle = nullptr;
  command->set_speed = false;
  command->advance = false;
  constexpr char kAdvanceKey[] = "advance\",\"budget_game_ms\":";
  constexpr char kFrameKey[] = ",\"frame_game_ms\":";
  constexpr char kHealthKey[] = ",\"health_change_fraction\":";
  if (tail.rfind(kAdvanceKey, 0) == 0) {
    size_t at = sizeof(kAdvanceKey) - 1, digits = 0;
    uint32_t budget = 0;
    for (; at < tail.size() && tail[at] >= '0' && tail[at] <= '9'; ++at, ++digits) {
      budget = budget * 10 + static_cast<uint32_t>(tail[at] - '0');
      if (budget > kMaxAdvanceBudgetMillis) return false;
    }
    if (digits == 0 || budget < kMinAdvanceBudgetMillis ||
        tail.compare(at, sizeof(kFrameKey) - 1, kFrameKey) != 0) {
      return false;
    }
    at += sizeof(kFrameKey) - 1;
    const size_t frame_end = tail.find(',', at);
    double frame_millis = 0.0, health_fraction = 0.0;
    if (frame_end == std::string::npos ||
        !ParseNumber(tail.substr(at, frame_end - at), &frame_millis) ||
        frame_millis < kMinFrameGameMillis || frame_millis > kMaxFrameGameMillis ||
        tail.compare(frame_end, sizeof(kHealthKey) - 1, kHealthKey) != 0) {
      return false;
    }
    at = frame_end + sizeof(kHealthKey) - 1;
    if (tail.size() < at + 2 || tail.back() != '}' ||
        !ParseNumber(tail.substr(at, tail.size() - at - 1), &health_fraction) ||
        health_fraction < 0.0 || health_fraction > 1.0) {
      return false;
    }
    command->advance = true;
    command->budget_game_millis = budget;
    command->frame_game_millis = static_cast<float>(frame_millis);
    command->health_change_fraction = static_cast<float>(health_fraction);
    command->family = nullptr; command->index = 0; command->lifecycle = nullptr;
    return true;
  }
  if (tail.rfind("set_speed\",\"value\":", 0) == 0) {
    const std::string value = tail.substr(std::strlen("set_speed\",\"value\":"));
    double parsed = 0.0;
    if (value.size() < 2 || value.back() != '}' ||
        !ParseNumber(value.substr(0, value.size() - 1), &parsed) ||
        parsed < kMinRequestedSpeed || parsed > kMaxRequestedSpeed) {
      return false;
    }
    command->set_speed = true;
    command->speed = static_cast<float>(parsed);
    command->family = nullptr; command->index = 0; command->lifecycle = nullptr;
    return true;
  }
  if (tail.rfind("lifecycle\",\"action\":\"", 0) == 0) {
    const size_t action_start = std::strlen("lifecycle\",\"action\":\"");
    if (tail.size() < action_start + 3 || tail.substr(tail.size() - 2) != "\"}") return false;
    const std::string action = tail.substr(action_start, tail.size() - action_start - 2);
    for (const LifecycleAction& candidate : kLifecycleActions) {
      if (action == candidate.name) { command->lifecycle = &candidate; break; }
    }
    command->family = nullptr; command->index = 0;
    return command->lifecycle != nullptr;
  }
  const char* family = nullptr;
  if (tail.rfind("buy_upgrade\",\"family\":\"attack\",\"index\":", 0) == 0) family = "attack";
  else if (tail.rfind("buy_upgrade\",\"family\":\"defense\",\"index\":", 0) == 0) family = "defense";
  else if (tail.rfind("buy_upgrade\",\"family\":\"utility\",\"index\":", 0) == 0) family = "utility";
  else return false;
  const size_t digits = tail.rfind(':') + 1; if (tail.back() != '}' || digits >= tail.size() - 1) return false;
  command->index = 0; for (size_t i = digits; i + 1 < tail.size(); ++i) { if (tail[i] < '0' || tail[i] > '9') return false; command->index = command->index * 10 + (tail[i] - '0'); }
  command->family = family; return true;
}

// What one `advance` cost. Reported on every command result so the host's
// accounting of game time has a single shape to read; it is all zero for the
// commands that advance no frames.
struct AdvanceDetail {
  int32_t frames = 0;
  // Budget accounting: the frames stepped under the advance loop times
  // `frame_game_ms`, which is what the advance asked the world to be worth. The
  // tail frames the pause takes to land are counted in `frames` but not here:
  // they run at real-time pacing and were never asked to be worth that much.
  uint32_t game_millis = 0;
  // The game's own per-round clock, measured across the same advance. Reported
  // beside `game_millis` so the 1:1 mapping between them can be checked rather
  // than assumed. Zero whenever the settled state could not be read, which the
  // outcome and reason on the same result already say.
  uint32_t round_millis = 0;
  uint64_t wall_micros = 0;
};

bool SendCommandResult(int client, const Command& command, const char* outcome, const char* reason, uint64_t sequence, const AdvanceDetail& detail) {
  char payload[512];
  const int size = std::snprintf(payload, sizeof(payload), "{\"type\":\"command_result\",\"protocol_version\":1,\"request_id\":\"%s\",\"outcome\":\"%s\",\"reason\":\"%s\",\"observation_sequence\":%llu,\"frames\":%d,\"game_ms\":%u,\"round_ms\":%u,\"wall_micros\":%llu}", command.request_id, outcome, reason, static_cast<unsigned long long>(sequence), detail.frames, detail.game_millis, detail.round_millis, static_cast<unsigned long long>(detail.wall_micros));
  return size > 0 && static_cast<size_t>(size) < sizeof(payload) && SendFrame(client, payload);
}

bool IsLowerSha256(const char* value) {
  if (value == nullptr || std::strlen(value) != 64) return false;
  for (size_t index = 0; index < 64; ++index) {
    const char character = value[index];
    if (!((character >= '0' && character <= '9') || (character >= 'a' && character <= 'f'))) {
      return false;
    }
  }
  return true;
}

bool HasValidBuildCompatibility() {
  return std::strcmp(TOWER_BRIDGE_PACKAGE_VERSION, "unconfigured") != 0 &&
         TOWER_BRIDGE_PACKAGE_VERSION_CODE > 0 &&
         std::strcmp(TOWER_BRIDGE_VERSION, "") != 0 &&
         std::strcmp(TOWER_BRIDGE_PROFILE_ID, "unconfigured") != 0 &&
         IsLowerSha256(TOWER_BRIDGE_OFFICIAL_SIGNER_SHA256) &&
         IsLowerSha256(TOWER_BRIDGE_ORIGINAL_LIBUNITY_SHA256) &&
         IsLowerSha256(TOWER_BRIDGE_LIBIL2CPP_SHA256);
}

std::string Handshake(const Il2CppApi& api, const MainFields& fields) {
  float game_speed = 0.0F;
  api.field_static_get_value(fields.game_speed, &game_speed);
  char message[1152];
  std::snprintf(message, sizeof(message),
                "{\"type\":\"handshake\",\"protocol_version\":1,"
                "\"bridge_version\":\"%s\",\"mode\":\"instrumented_training\","
                "\"command_capability\":\"semantic-v2\",\"compatibility\":{"
                "\"package_version\":\"%s\",\"package_version_code\":%d,"
                "\"official_signer_sha256\":\"%s\","
                "\"original_libunity_sha256\":\"%s\",\"libil2cpp_sha256\":\"%s\","
                "\"unity_version\":\"%s\",\"il2cpp_metadata_version\":%d,"
                "\"profile_id\":\"%s\"},"
                "\"game_speed\":%.9g}",
                TOWER_BRIDGE_VERSION, TOWER_BRIDGE_PACKAGE_VERSION,
                TOWER_BRIDGE_PACKAGE_VERSION_CODE, TOWER_BRIDGE_OFFICIAL_SIGNER_SHA256,
                TOWER_BRIDGE_ORIGINAL_LIBUNITY_SHA256, TOWER_BRIDGE_LIBIL2CPP_SHA256,
                TOWER_BRIDGE_UNITY_VERSION, TOWER_BRIDGE_METADATA_VERSION,
                TOWER_BRIDGE_PROFILE_ID, game_speed);
  return message;
}

int OpenLoopbackServer() {
  const int server = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
  if (server < 0) return -1;
  int reuse = 1;
  setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  address.sin_port = htons(kLoopbackPort);
  if (bind(server, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) != 0 ||
      listen(server, 1) != 0) {
    close(server);
    return -1;
  }
  return server;
}

// Engine accessors resolved once and validated by provenance: a pointer we
// cannot attribute to libunity.so is not called at all. These are leaf bindings -
// they read or write one field of a manager singleton and allocate nothing -
// which is what makes them safe from this thread, unlike managed game code.
struct EngineClock {
  int32_t (*get_frame_count)() = nullptr;
  void (*set_capture_delta)(float) = nullptr;
  bool resolved = false;
};

void* ResolveEngineIcall(const char* signature) {
  void* il2cpp = dlopen("libil2cpp.so", RTLD_NOW | RTLD_NOLOAD);
  void* (*resolve_icall)(const char*) = nullptr;
  if (il2cpp == nullptr || !Resolve(il2cpp, "il2cpp_resolve_icall", &resolve_icall)) return nullptr;
  void* pointer = resolve_icall(signature);
  Dl_info info{};
  // Refuse anything that does not come from the engine: a pointer we cannot
  // attribute is not worth calling blind.
  if (pointer == nullptr || dladdr(pointer, &info) == 0 || info.dli_fname == nullptr ||
      std::strstr(info.dli_fname, "libunity.so") == nullptr) {
    return nullptr;
  }
  return pointer;
}

const EngineClock& Clock() {
  static EngineClock clock;
  if (!clock.resolved) {
    clock.resolved = true;
    clock.get_frame_count =
        reinterpret_cast<int32_t (*)()>(ResolveEngineIcall("UnityEngine.Time::get_frameCount()"));
    clock.set_capture_delta = reinterpret_cast<void (*)(float)>(
        ResolveEngineIcall("UnityEngine.Time::set_captureDeltaTime(System.Single)"));
    __android_log_print(ANDROID_LOG_INFO, kLogTag, "clock frames=%p set_capture=%p",
                        reinterpret_cast<void*>(clock.get_frame_count),
                        reinterpret_cast<void*>(clock.set_capture_delta));
  }
  return clock;
}

#ifdef TOWER_BRIDGE_DIAGNOSTICS
Il2CppClass* g_diagnostic_main = nullptr;

// Sample candidate in-run clock fields so the real one is identified by evidence
// rather than by assuming the first name that looks right.
void LogClockCandidates(const Il2CppApi& api, Il2CppClass* main) {
  static const char* kCandidates[] = {"realTimeThisRound", "gameplayTimeThisRound", "roundTime",
                                      "playTime"};
  Il2CppObject* instance = nullptr;
  FieldInfo* instance_field = api.class_get_field_from_name(main, "<Instance>k__BackingField");
  if (instance_field == nullptr) return;
  api.field_static_get_value(instance_field, &instance);
  if (instance == nullptr) return;
  for (const char* name : kCandidates) {
    FieldInfo* field = api.class_get_field_from_name(main, name);
    if (field == nullptr) {
      __android_log_print(ANDROID_LOG_INFO, kLogTag, "clock %s absent", name);
      continue;
    }
    double as_double = 0.0;
    float as_float = 0.0F;
    api.field_get_value(instance, field, &as_double);
    api.field_get_value(instance, field, &as_float);
    __android_log_print(ANDROID_LOG_INFO, kLogTag, "clock %s double=%.4f float=%.4f", name,
                        as_double, static_cast<double>(as_float));
  }
}

// Report whether `Main.Instance` is a live component or Unity's fake null. This
// is the decisive test of whether the lifecycle receiver exists outside the
// battle scene: a zeroed native handle means the GameObject is gone, which is
// why `UnitySendMessage` has nothing to deliver to.
void LogMainLiveness(const Il2CppApi& api, const MainFields& fields, const char* when) {
  Il2CppObject* main = nullptr;
  api.field_static_get_value(fields.instance, &main);
  void* handle = nullptr;
  if (main != nullptr) {
    Il2CppClass* klass = api.object_get_class(main);
    FieldInfo* cached =
        klass == nullptr ? nullptr : api.class_get_field_from_name(klass, "m_CachedPtr");
    if (cached != nullptr) api.field_get_value(main, cached, &handle);
    const EngineClock& clock = Clock();
    __android_log_print(ANDROID_LOG_INFO, kLogTag,
                        "liveness %s managed=%p cached_ptr=%p field=%s frames=%d", when,
                        static_cast<void*>(main), handle, cached ? "found" : "missing",
                        clock.get_frame_count == nullptr ? -1 : clock.get_frame_count());
    return;
  }
  __android_log_print(ANDROID_LOG_INFO, kLogTag, "liveness %s managed=null", when);
}
#endif

// Every emitted state message advances one monotonic sequence, so a command can
// always bind the state it was decided from, including between episodes.
bool SendState(int client, const Il2CppApi& api, const MainFields& fields, uint64_t sequence) {
  std::string payload;
  switch (BuildObservation(api, fields, sequence, &payload)) {
    case ObservationResult::kOk:
#ifdef TOWER_BRIDGE_DIAGNOSTICS
      LogClockCandidates(api, g_diagnostic_main);
#endif
      return SendFrame(client, payload);
    case ObservationResult::kNoRun:
#ifdef TOWER_BRIDGE_DIAGNOSTICS
      // Logged exactly where the receiver question is decided: if the handle is
      // zero here, `Main` is destroyed and UnitySendMessage has no target.
      LogMainLiveness(api, fields, "no_run");
#endif
      return SendFrame(client, "{\"type\":\"run_unavailable\",\"sequence\":" +
                                   std::to_string(sequence) + ",\"reason\":\"no_initialized_run\"}");
    case ObservationResult::kError:
      break;
  }
  SendError(client, "observation_error", "exact snapshot could not be encoded");
  return false;
}

void RefreshCosts() {
  UnitySendMessage send = ResolveUnitySendMessage();
  if (send == nullptr) return;
  for (const char* method : kCostRefreshMethods) {
    send(TOWER_BRIDGE_MAIN_GAME_OBJECT, method, "");
  }
}

float CurrentGameSpeed(const Il2CppApi& api, const MainFields& fields) {
  float speed = 0.0F;
  api.field_static_get_value(fields.game_speed, &speed);
  return (std::isfinite(speed) && speed > 1.0F) ? speed : 1.0F;
}

// Real elapsed microseconds. `usleep` is a floor, not a promise, so counting
// sleeps overstates progress and understates cost; the advance ceiling is only
// a ceiling if it is measured.
uint64_t MonotonicMicros() {
  timespec now{};
  clock_gettime(CLOCK_MONOTONIC, &now);
  return static_cast<uint64_t>(now.tv_sec) * 1000000ULL +
         static_cast<uint64_t>(now.tv_nsec) / 1000ULL;
}

useconds_t GameTimeInterval(useconds_t interval, float speed) {
  const auto scaled = static_cast<useconds_t>(static_cast<float>(interval) / speed);
  return scaled < kMinIntervalMicros ? kMinIntervalMicros : scaled;
}

bool RunIsActive(const Il2CppApi& api, const MainFields& fields) {
  Il2CppObject* main = nullptr;
  api.field_static_get_value(fields.instance, &main);
  if (!NativeHandleIsAlive(api, main)) return false;
  uint8_t game_over = 0, round_active = 0;
  double health = 0.0;
  if (!ReadField(api, main, fields.game_over, &game_over) ||
      !ReadField(api, main, fields.round_active, &round_active) ||
      !ReadField(api, main, fields.tower_health, &health) || !std::isfinite(health)) {
    return false;
  }
  return round_active == 1 && game_over == 0;
}

// Exactly the inputs the host's decision predicate reads, sampled together. The
// bridge does not own the predicate - Python does - so this mirrors
// `RunStateBuilder` and `_events_between` member for member, including the
// `max_health <= 0` case and the clamp, and diverging from them is a defect.
struct DecisionSnapshot {
  bool active = false;
  int32_t wave = 0;
  double health_fraction = 0.0;
  //: The game's own per-round clock, so an advance can report the game time it
  //: really passed beside the game time it budgeted for.
  double round_time = 0.0;
  bool available[3 * kMaskSlotsPerFamily] = {};
};

// A slot the game does not report is unavailable, never assumed cheap.
void ReadFamilyAvailability(const Il2CppApi& api, Il2CppObject* main, const FamilyFields& family,
                            double cash, bool active, bool* available) {
  Il2CppArray *costs = nullptr, *unlocked = nullptr, *maxed = nullptr;
  if (!ReadField(api, main, family.cost, &costs) ||
      !ReadField(api, main, family.unlocked, &unlocked) ||
      !ReadField(api, main, family.maxed, &maxed) || costs == nullptr || unlocked == nullptr ||
      maxed == nullptr) {
    return;
  }
  const size_t count = api.array_length(costs);
  if (api.array_length(unlocked) != count || api.array_length(maxed) != count) return;
  for (size_t index = 0; index < kMaskSlotsPerFamily && index < count; ++index) {
    double cost = 0.0;
    uint8_t is_unlocked = 0, is_maxed = 0;
    if (!ReadPrimitiveArray(api, costs, index, &cost) ||
        !ReadPrimitiveArray(api, unlocked, index, &is_unlocked) ||
        !ReadPrimitiveArray(api, maxed, index, &is_maxed) || !std::isfinite(cost)) {
      continue;
    }
    available[index] = active && is_unlocked == 1 && is_maxed == 0 && cost > 0.0 && cost <= cash;
  }
}

// Fails when `Main` is gone or unreadable, which the caller treats as the run
// having ended - the same conclusion the host draws from an unreadable state.
bool ReadDecisionSnapshot(const Il2CppApi& api, const MainFields& fields, DecisionSnapshot* snapshot) {
  Il2CppObject* main = nullptr;
  api.field_static_get_value(fields.instance, &main);
  if (!NativeHandleIsAlive(api, main)) return false;
  double cash = 0.0, health = 0.0, max_health = 0.0;
  uint8_t game_over = 0, round_active = 0;
  if (!ReadField(api, main, fields.cash, &cash) ||
      !ReadField(api, main, fields.current_wave, &snapshot->wave) ||
      !ReadField(api, main, fields.tower_health, &health) ||
      !ReadField(api, main, fields.tower_max_health, &max_health) ||
      !ReadField(api, main, fields.game_over, &game_over) ||
      !ReadField(api, main, fields.round_active, &round_active) ||
      !std::isfinite(cash) || !std::isfinite(health) || !std::isfinite(max_health)) {
    return false;
  }
  // The game stores the round clock as a single, so it has to be read as one:
  // reading it into a double returns garbage rather than the time.
  float round_time = 0.0F;
  if (!ReadField(api, main, fields.round_time, &round_time) || !std::isfinite(round_time)) {
    return false;
  }
  snapshot->round_time = round_time;
  snapshot->active = round_active == 1 && game_over == 0;
  if (max_health <= 0.0) {
    snapshot->health_fraction = 0.0;
  } else {
    const double ratio = health / max_health;
    snapshot->health_fraction = ratio < 0.0 ? 0.0 : (ratio > 1.0 ? 1.0 : ratio);
  }
  ReadFamilyAvailability(api, main, fields.attack, cash, snapshot->active, snapshot->available);
  ReadFamilyAvailability(api, main, fields.defense, cash, snapshot->active,
                         snapshot->available + kMaskSlotsPerFamily);
  ReadFamilyAvailability(api, main, fields.utility, cash, snapshot->active,
                         snapshot->available + 2 * kMaskSlotsPerFamily);
  return true;
}

bool BecameAvailable(const DecisionSnapshot& before, const DecisionSnapshot& after) {
  // `WAIT` is the host's mask entry zero and is exactly run-is-active.
  if (after.active && !before.active) return true;
  for (size_t index = 0; index < 3 * kMaskSlotsPerFamily; ++index) {
    if (after.available[index] && !before.available[index]) return true;
  }
  return false;
}

// Advance the world frame by frame until something worth deciding about happens,
// or until the game-time budget is spent, then pause again.
//
// `captureDeltaTime` makes every frame rendered under the loop worth exactly
// `frame_game_ms` of game time however long it took to render, so the game time
// between decisions depends on neither the game's speed multiplier nor on how fast this host is.
// Because the loop lives here rather than in the host, the policy's own latency
// costs no game time at all and one decision costs one round trip.
//
// Returns false only when the client connection is gone. `*paused` reports
// whether this advance left the world paused, which only this function knows:
// it is the one that decides whether to press `Pause`. Deriving it afterwards
// from `RunIsActive` would read a different instant, and a round that started
// in between would then be reported as a paused world - the one way the host
// could be left acting on a view the bridge had stopped refreshing.
bool AdvanceUntilEvent(int client, const Il2CppApi& api, const MainFields& fields,
                       const Command& command, uint64_t sequence, const char** outcome,
                       const char** reason, AdvanceDetail* detail, bool* paused) {
  *paused = false;
  const EngineClock& clock = Clock();
  UnitySendMessage send = ResolveUnitySendMessage();
  if (clock.get_frame_count == nullptr || clock.set_capture_delta == nullptr || send == nullptr) {
    *outcome = "ambiguous";
    *reason = "clock_unavailable";
    return true;
  }
  // The world is paused between commands, so entry state is a settled reading.
  DecisionSnapshot before{}, after{}, settled{};
  if (!ReadDecisionSnapshot(api, fields, &before) || !before.active) {
    *reason = "event:run_ended";
    return true;
  }
  const double health_threshold = static_cast<double>(command.health_change_fraction);
  clock.set_capture_delta(command.frame_game_millis / 1000.0F);
  int32_t last_count = clock.get_frame_count();
  send(TOWER_BRIDGE_MAIN_GAME_OBJECT, "Unpause", "");

  const uint64_t started = MonotonicMicros();
  uint64_t last_heartbeat = started;
  double game_millis = 0.0;
  bool connected = true;
#ifdef TOWER_BRIDGE_DIAGNOSTICS
  // The round clock at three moments of the advance, so the difference between
  // the game's clock and this loop's frame arithmetic can be located rather
  // than inferred. No extra reads: these reuse snapshots taken anyway.
  const float probe_t0 = before.round_time;
  float probe_t1 = before.round_time;
#endif
  while (true) {
    usleep(kFramePollMicros);
    const uint64_t now = MonotonicMicros();
    detail->wall_micros = now - started;
    const int32_t count = clock.get_frame_count();
    if (count > last_count) {
      const int32_t rendered = count - last_count;
      last_count = count;
      detail->frames += rendered;
      game_millis += rendered * static_cast<double>(command.frame_game_millis);
      // These mid-frame readings decide only when to stop. What the advance
      // reports is decided further down, from the settled state.
      const bool read = ReadDecisionSnapshot(api, fields, &after);
#ifdef TOWER_BRIDGE_DIAGNOSTICS
      if (read) probe_t1 = after.round_time;
#endif
      if (!read || !after.active) break;
      if (after.wave != before.wave) break;
      if (BecameAvailable(before, after)) break;
      if (std::fabs(after.health_fraction - before.health_fraction) >= health_threshold) break;
      if (game_millis >= static_cast<double>(command.budget_game_millis)) break;
    }
    if (detail->wall_micros >= kAdvanceWallBudgetMicros) break;
    // A long advance must keep proving the bridge is alive, or the host cannot
    // tell a quiet world from a dead connection.
    if (now - last_heartbeat >= kHeartbeatIntervalMicros) {
      last_heartbeat = now;
      if (!SendFrame(client, "{\"type\":\"heartbeat\",\"last_observation_sequence\":" +
                                 std::to_string(sequence) + "}")) {
        connected = false;
        break;
      }
    }
  }

#ifdef TOWER_BRIDGE_DIAGNOSTICS
  const int32_t probe_loop_frames = detail->frames;
#endif
  // Pausing a run that has already ended would press a control the game no
  // longer owns a receiver for; RunIsActive is false then anyway.
  if (RunIsActive(api, fields)) {
    *paused = true;
    // `Pause` is dispatched to the main thread and lands a frame or two later.
    // Those tail frames advance no meaningful world state, so pacing them at
    // `frame_game_ms` each credited the budget with game time the world never
    // simulated. They run at real-time pacing instead, and are counted as
    // frames but not as game time; what they are really worth is small, and the
    // round clock does measure it.
    clock.set_capture_delta(0.0F);
    send(TOWER_BRIDGE_MAIN_GAME_OBJECT, "Pause", "");
    const uint64_t settle_started = MonotonicMicros();
    const int32_t settle_from = last_count;
    while (last_count - settle_from < kPauseSettleFrames) {
      usleep(kFramePollMicros);
      const int32_t count = clock.get_frame_count();
      if (count > last_count) {
        const int32_t rendered = count - last_count;
        last_count = count;
        detail->frames += rendered;
      }
      if (MonotonicMicros() - settle_started >= kPauseSettleMicros) break;
    }
    detail->wall_micros = MonotonicMicros() - started;
  }

  // The settled state is both what `SendState` will emit to the host and what
  // this result describes, so the two can never disagree. The loop above chose
  // the moment to stop; this chooses what that moment turned out to be.
  const bool readable = ReadDecisionSnapshot(api, fields, &settled);
  bool ended = false;
  if (!readable || !settled.active) {
    *reason = "event:run_ended";
    ended = true;
  } else if (settled.wave != before.wave) {
    *reason = "event:wave_changed";
  } else if (BecameAvailable(before, settled)) {
    *reason = "event:newly_affordable";
  } else if (std::fabs(settled.health_fraction - before.health_fraction) >= health_threshold) {
    *reason = "event:health_changed";
  } else {
    *reason = "budget_exhausted";
  }
  // Restore real-time pacing so nothing outside an advance observes a stopped
  // clock. Already done above when the run was still active and was paused;
  // this covers the path where it had ended.
  clock.set_capture_delta(0.0F);
  detail->game_millis = static_cast<uint32_t>(game_millis + 0.5);
  // An unreadable settled state means the run ended under the advance, which
  // the outcome reports; there is no round clock left to difference. The round
  // clock also resets when a round ends, so a negative difference is not a
  // measurement either.
  const double round_millis = readable ? (settled.round_time - before.round_time) * 1000.0 : 0.0;
  detail->round_millis =
      round_millis > 0.0 ? static_cast<uint32_t>(round_millis + 0.5) : 0U;
  if (detail->frames == 0 && !ended) {
    *outcome = "ambiguous";
    *reason = "no_frame_rendered";
  }
#ifdef TOWER_BRIDGE_DIAGNOSTICS
  // One line per advance, fixed field order, so a device run can be parsed
  // without guessing: the round clock before `Unpause`, at the loop's last
  // reading, and after the settle window.
  __android_log_print(ANDROID_LOG_INFO, kLogTag,
                      "clockprobe t0=%.3f t1=%.3f t2=%.3f loop_frames=%d settle_frames=%d "
                      "frame_game_ms=%.1f wall_us=%llu",
                      static_cast<double>(probe_t0), static_cast<double>(probe_t1),
                      static_cast<double>(readable ? settled.round_time : probe_t1),
                      probe_loop_frames, detail->frames - probe_loop_frames,
                      static_cast<double>(command.frame_game_millis),
                      static_cast<unsigned long long>(detail->wall_micros));
#endif
  return connected;
}

#ifdef TOWER_BRIDGE_DIAGNOSTICS
// Resolve the engine accessors the frame-exact step would need, and report which
// of them exist and which library they come from. Nothing is called: this
// separates "the binding is unreachable" from "the call is unsafe", which are
// different problems with different answers.
void LogEngineIcalls() {
  void* il2cpp = dlopen("libil2cpp.so", RTLD_NOW | RTLD_NOLOAD);
  if (il2cpp == nullptr) return;
  void* (*resolve_icall)(const char*) = nullptr;
  if (!Resolve(il2cpp, "il2cpp_resolve_icall", &resolve_icall)) {
    __android_log_print(ANDROID_LOG_INFO, kLogTag, "icall il2cpp_resolve_icall missing");
    return;
  }
  static const char* kNames[] = {
      "UnityEngine.Time::get_frameCount()",
      "UnityEngine.Time::get_frameCount",
      "UnityEngine.Time::get_captureDeltaTime()",
      "UnityEngine.Time::set_captureDeltaTime(System.Single)",
      "UnityEngine.Time::set_captureDeltaTime",
      "UnityEngine.Time::get_timeScale()",
      "UnityEngine.Time::get_fixedDeltaTime()",
      "UnityEngine.Time::get_maximumDeltaTime()",
      "UnityEngine.Application::set_targetFrameRate(System.Int32)",
      "UnityEngine.QualitySettings::set_vSyncCount(System.Int32)",
      "UnityEngine.Object::GetName(UnityEngine.Object)",
  };
  for (const char* name : kNames) {
    void* pointer = resolve_icall(name);
    Dl_info info{};
    const bool located = pointer != nullptr && dladdr(pointer, &info) != 0;
    __android_log_print(ANDROID_LOG_INFO, kLogTag, "icall %s -> %p (%s)", name, pointer,
                        located && info.dli_fname ? info.dli_fname : "unattributed");
  }
  // Presence only: a stop-the-world heap walk is the fallback if no receiver is
  // found, and knowing now whether the exports exist costs nothing.
  void* stop_world = dlsym(il2cpp, "il2cpp_stop_gc_world");
  void* foreach_heap = dlsym(il2cpp, "il2cpp_gc_foreach_heap");
  __android_log_print(ANDROID_LOG_INFO, kLogTag, "gc_world stop=%p foreach=%p", stop_world,
                      foreach_heap);
  // `maximumDeltaTime` is the engine's hard ceiling on how much time one frame
  // may advance, so it is the ceiling on `frame_game_ms`. Read once, here, where
  // the value lands in logcat on deploy instead of being assumed.
  auto get_maximum_delta = reinterpret_cast<float (*)()>(
      ResolveEngineIcall("UnityEngine.Time::get_maximumDeltaTime()"));
  auto get_fixed_delta = reinterpret_cast<float (*)()>(
      ResolveEngineIcall("UnityEngine.Time::get_fixedDeltaTime()"));
  if (get_maximum_delta != nullptr && get_fixed_delta != nullptr) {
    __android_log_print(ANDROID_LOG_INFO, kLogTag, "engine maximum_delta=%.6f fixed_delta=%.6f",
                        static_cast<double>(get_maximum_delta()),
                        static_cast<double>(get_fixed_delta()));
  }
}

// Locate a semantic entry point that does not live on `Main`, by scanning every
// class for method names containing an allowlisted fragment.
void LogMatchingMethods(const Il2CppApi& api, Il2CppDomain* domain) {
  void* il2cpp = dlopen("libil2cpp.so", RTLD_NOW | RTLD_NOLOAD);
  if (il2cpp == nullptr || domain == nullptr) return;
  const MethodInfo* (*class_get_methods)(Il2CppClass*, void**) = nullptr;
  const char* (*method_get_name)(const MethodInfo*) = nullptr;
  uint32_t (*method_get_param_count)(const MethodInfo*) = nullptr;
  if (!Resolve(il2cpp, "il2cpp_class_get_methods", &class_get_methods) ||
      !Resolve(il2cpp, "il2cpp_method_get_name", &method_get_name) ||
      !Resolve(il2cpp, "il2cpp_method_get_param_count", &method_get_param_count)) {
    return;
  }
  static const char* kFragments[] = {"Retry", "Restart", "GameEnd", "NewRound", "Respawn"};
  size_t assemblies_count = 0;
  const Il2CppAssembly** assemblies = api.domain_get_assemblies(domain, &assemblies_count);
  for (size_t a = 0; a < assemblies_count; ++a) {
    const Il2CppImage* image = api.assembly_get_image(assemblies[a]);
    for (size_t c = 0; c < api.image_get_class_count(image); ++c) {
      Il2CppClass* klass = api.image_get_class(image, c);
      const char* class_name = klass == nullptr ? nullptr : api.class_get_name(klass);
      if (class_name == nullptr) continue;
      void* iterator = nullptr;
      for (const MethodInfo* method = class_get_methods(klass, &iterator); method != nullptr;
           method = class_get_methods(klass, &iterator)) {
        const char* name = method_get_name(method);
        if (name == nullptr) continue;
        for (const char* fragment : kFragments) {
          if (std::strstr(name, fragment) != nullptr) {
            __android_log_print(ANDROID_LOG_INFO, kLogTag, "scan %s.%s/%u", class_name, name,
                                method_get_param_count(method));
            break;
          }
        }
      }
    }
  }
}
#endif

void ServeClient(int client, const Il2CppApi& api, const MainFields& fields) {
  if (!HasValidBuildCompatibility()) {
    SendError(client, "compatibility_error", "bridge build compatibility is not configured");
    return;
  }
  if (ResolveUnitySendMessage() == nullptr) {
    SendError(client, "compatibility_error", "UnitySendMessage is unavailable");
    return;
  }
  if (!SendFrame(client, Handshake(api, fields))) return;
  uint64_t sequence = 0;
  char last_request_id[65] = "";
  useconds_t heartbeat_elapsed = 0;
  // This bridge is the only thing that pauses the world, so it knows when the
  // world is paused. That matters because the sequence exists to stop the host
  // acting on a stale view: while the world is paused no new information can
  // exist, so the view cannot go stale and the sequence must not move.
  bool world_paused = false;
  while (true) {
    fd_set readable;
    FD_ZERO(&readable);
    FD_SET(client, &readable);
    const useconds_t wait_micros =
        GameTimeInterval(kObservationIntervalMicros, CurrentGameSpeed(api, fields));
    timeval wait{};
    wait.tv_sec = static_cast<time_t>(wait_micros / 1000000);
    wait.tv_usec = static_cast<suseconds_t>(wait_micros % 1000000);
    if (select(client + 1, &readable, nullptr, nullptr, &wait) > 0) {
      std::string payload;
      Command command{};
      if (!ReadInboundFrame(client, &payload) || !ParseCommand(payload, &command)) {
        SendError(client, "protocol_error", "malformed command"); return;
      }
      if (command.expected_sequence != sequence || std::strcmp(command.request_id, last_request_id) == 0) {
        SendCommandResult(client, command, "rejected", "stale_or_duplicate", sequence, AdvanceDetail{}); continue;
      }
      UpgradeEvidence before{}, after{};
      const char* outcome = "confirmed";
      const char* reason = "confirmed_state_change";
      AdvanceDetail detail{};
      if (command.advance) {
        // One round trip per decision: the bridge advances frames of fixed game
        // time until the host's own decision predicate would fire.
        reason = "budget_exhausted";
        // The advance pauses the world again exactly when the run is still
        // active; a run that ended under it was never paused and its screens
        // keep changing, so its state must keep streaming. It reports that
        // itself rather than being asked again afterwards.
        if (!AdvanceUntilEvent(client, api, fields, command, sequence, &outcome, &reason,
                               &detail, &world_paused)) {
          return;
        }
      } else if (command.set_speed) {
        float applied = 0.0F;
        api.field_static_set_value(fields.game_speed, &command.speed);
        ResolveUnitySendMessage()(TOWER_BRIDGE_MAIN_GAME_OBJECT, "GameSpeedModifier", "");
        bool settled = false;
        for (useconds_t elapsed = 0; elapsed < kCommandTimeoutMicros;
             elapsed += kCommandPollMicros) {
          usleep(kCommandPollMicros);
          api.field_static_get_value(fields.game_speed, &applied);
          if (std::isfinite(applied) && std::fabs(applied - command.speed) < 0.01F) {
            settled = true;
            break;
          }
        }
        outcome = settled ? "confirmed" : "rejected";
        reason = settled ? "speed_applied" : "speed_not_applied";
      } else if (command.lifecycle != nullptr) {
        // The game owns the transition; the bridge only presses its own control
        // and then waits for the game's own state to agree.
        ResolveUnitySendMessage()(TOWER_BRIDGE_MAIN_GAME_OBJECT, command.lifecycle->method, "");
        reason = command.lifecycle->expect_active ? "run_active" : "run_closed";
        bool settled = false;
        // A scene transition takes seconds. The stream must keep proving it is
        // alive, or the host cannot tell a slow transition from a dead bridge.
        useconds_t since_heartbeat = 0;
        for (useconds_t elapsed = 0; elapsed < kLifecycleTimeoutMicros;
             elapsed += kLifecyclePollMicros) {
          usleep(kLifecyclePollMicros);
          if (RunIsActive(api, fields) == command.lifecycle->expect_active) { settled = true; break; }
          since_heartbeat += kLifecyclePollMicros;
          if (since_heartbeat >= kHeartbeatIntervalMicros) {
            since_heartbeat = 0;
            if (!SendFrame(client, "{\"type\":\"heartbeat\",\"last_observation_sequence\":" +
                                       std::to_string(sequence) + "}")) return;
          }
        }
        // `pause` is the one lifecycle control that stops the world; every other
        // one leaves it running, including the `unpause` that ends a session.
        // Only a confirmed outcome may hold the sequence: a `pause` pressed from
        // a screen that cannot take it times out ambiguous, and believing it
        // would freeze the stream for a world that never stopped.
        world_paused = settled && std::strcmp(command.lifecycle->name, "pause") == 0;
        if (settled) {
          if (command.lifecycle->expect_active) RefreshCosts();
        } else {
          outcome = "ambiguous"; reason = "lifecycle_timeout";
        }
      } else {
        // `unlocked` is the in-run availability the game itself offers. Live 29.0.3
        // evidence shows `tier_unlocked` is false for every offered upgrade, so it
        // is reported state, not a purchase precondition.
        // A non-positive cost is never an offered price: early in a run the game
        // has not yet populated every family's cost array, and a zero there must
        // not be mistaken for a free purchase.
        if (!ReadUpgradeEvidence(api, fields, command.family, command.index, &before) ||
            !before.unlocked || before.maxed || before.cost <= 0.0 || before.cash < before.cost) {
          outcome = "rejected"; reason = "precondition_failed";
        } else {
          int32_t selection = static_cast<int32_t>(command.index);
          api.field_static_set_value(fields.upgrade_select, &selection);
          const char* method = std::strcmp(command.family, "attack") == 0 ? "UpgradeButton" :
              (std::strcmp(command.family, "defense") == 0 ? "UpgradeDefenseButton" : "UpgradeUtilityButton");
          ResolveUnitySendMessage()(TOWER_BRIDGE_MAIN_GAME_OBJECT, method, "");
          bool confirmed = false;
          // The game owns the purchase. Only its own level increment confirms one.
          // In-run cash rises continuously from kills, so a cash change alone is
          // neither confirmation nor contradiction.
          for (useconds_t elapsed = 0; elapsed < kCommandTimeoutMicros; elapsed += kCommandPollMicros) {
            usleep(kCommandPollMicros);
            if (!ReadUpgradeEvidence(api, fields, command.family, command.index, &after)) break;
            if (after.level == before.level + 1 && after.unlocked) { confirmed = true; break; }
            if (after.level < before.level || after.level > before.level + 1 ||
                !after.unlocked) { outcome = "ambiguous"; reason = "contradictory_state_change"; break; }
          }
          if (!confirmed && std::strcmp(outcome, "ambiguous") != 0) { outcome = "ambiguous"; reason = "confirmation_timeout"; }
        }
      }
      std::strncpy(last_request_id, command.request_id, sizeof(last_request_id) - 1);
      if (std::strcmp(outcome, "confirmed") == 0 && command.family != nullptr) RefreshCosts();
      if (!SendState(client, api, fields, ++sequence)) return;
      if (!SendCommandResult(client, command, outcome, reason, sequence, detail)) return;
      continue;
    }
    if (world_paused) {
      // A paused world has nothing new to report, and reporting it anyway would
      // move the sequence out from under a command the host is already
      // composing: a policy that thinks for longer than this interval would have
      // every command rejected as `stale_or_duplicate`. The heartbeat still
      // proves the bridge is alive and names the sequence that still stands.
      if (!SendFrame(client, "{\"type\":\"heartbeat\",\"last_observation_sequence\":" +
                                 std::to_string(sequence) + "}")) return;
      heartbeat_elapsed = 0;
      continue;
    }
    if (!SendState(client, api, fields, ++sequence)) return;
    heartbeat_elapsed += wait_micros;
    if (heartbeat_elapsed >= kHeartbeatIntervalMicros) {
      heartbeat_elapsed = 0;
      if (!SendFrame(client, "{\"type\":\"heartbeat\",\"last_observation_sequence\":" +
                                 std::to_string(sequence) + "}")) return;
    }
  }
}

// Resolving IL2CPP is deliberately deferred to the first client connection.
// `libil2cpp.so` is loadable long before its runtime is usable, and enumerating
// classes too early crashes the game process. A connecting host client is the
// evidence that the game has had time to initialize. The result is cached, so a
// reconnect never repeats the stabilization delay.
bool InitializeRuntime(Il2CppApi* api, MainFields* fields) {
  bool api_ready = false;
  for (int retries = 0; retries < 120 && !api_ready; ++retries) {
    api_ready = ResolveApi(api);
    if (!api_ready) usleep(250000);
  }
  if (!api_ready) {
    __android_log_print(ANDROID_LOG_ERROR, kLogTag, "IL2CPP exports are unavailable");
    return false;
  }
  usleep(kIl2CppInitializationDelayMicros);
  __android_log_print(ANDROID_LOG_INFO, kLogTag, "resolving domain");
  Il2CppDomain* domain = api->domain_get();
  __android_log_print(ANDROID_LOG_INFO, kLogTag, "domain=%d; attaching thread", domain != nullptr);
  Il2CppThread* thread = domain == nullptr ? nullptr : api->thread_attach(domain);
  __android_log_print(ANDROID_LOG_INFO, kLogTag, "thread=%d; finding Main", thread != nullptr);
  Il2CppClass* main = domain == nullptr ? nullptr : FindMainClass(*api, domain);
  __android_log_print(ANDROID_LOG_INFO, kLogTag, "main=%d; finding IntSelect", main != nullptr);
  Il2CppClass* int_select = domain == nullptr ? nullptr : FindClass(*api, domain, "IntSelect");
  __android_log_print(ANDROID_LOG_INFO, kLogTag, "int_select=%d; loading fields", int_select != nullptr);
  if (thread == nullptr || main == nullptr || int_select == nullptr ||
      !LoadFields(*api, main, int_select, fields)) {
    __android_log_print(ANDROID_LOG_ERROR, kLogTag,
                        "IL2CPP runtime is not ready: domain=%d thread=%d main=%d int_select=%d",
                        domain != nullptr, thread != nullptr, main != nullptr,
                        int_select != nullptr);
    return false;
  }
  __android_log_print(ANDROID_LOG_INFO, kLogTag, "IL2CPP runtime resolved");
#ifdef TOWER_BRIDGE_DIAGNOSTICS
  g_diagnostic_main = main;
  LogClassMembers(main, "Main");
  LogClassMembers(int_select, "IntSelect");
  LogMatchingMethods(*api, domain);
  LogEngineIcalls();
#endif
  return true;
}

void* BridgeThread(void*) {
  const int server = OpenLoopbackServer();
  if (server < 0) {
    __android_log_print(ANDROID_LOG_ERROR, kLogTag, "cannot bind loopback port %u", kLoopbackPort);
    return nullptr;
  }
  Il2CppApi api{};
  MainFields fields{};
  bool runtime_ready = false;
  while (true) {
    const int client = accept4(server, nullptr, nullptr, SOCK_CLOEXEC);
    if (client < 0) continue;
    if (!runtime_ready) runtime_ready = InitializeRuntime(&api, &fields);
    if (!runtime_ready) {
      SendError(client, "compatibility_error", "required IL2CPP exports unavailable");
      close(client);
      continue;
    }
    ServeClient(client, api, fields);
    close(client);
  }
}

void StartBridgeThread() {
  pthread_t thread;
  if (pthread_create(&thread, nullptr, BridgeThread, nullptr) != 0) {
    __android_log_print(ANDROID_LOG_ERROR, kLogTag, "cannot create bridge thread");
    return;
  }
  pthread_detach(thread);
}

pthread_once_t g_start_once = PTHREAD_ONCE_INIT;

}  // namespace

extern "C" __attribute__((constructor)) void TowerBridgeConstructor() {
  pthread_once(&g_start_once, StartBridgeThread);
}

extern "C" __attribute__((visibility("default"))) int JNI_OnLoad(void*, void*) {
  pthread_once(&g_start_once, StartBridgeThread);
  return 0x00010006;  // JNI_VERSION_1_6
}
