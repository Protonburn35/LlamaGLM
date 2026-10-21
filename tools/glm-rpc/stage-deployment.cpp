#include "stage-deployment.h"

#include <nlohmann/json.hpp>

#include <charconv>
#include <fstream>
#include <set>

namespace glm_rpc {
namespace {
using json = nlohmann::json;

void require(bool value, const char * reason) { if (!value) { throw std::invalid_argument(reason); } }

void fields(const json & object, std::initializer_list<const char *> allowed, size_t minimum) {
    require(object.is_object() && object.size() >= minimum,"invalid deployment object");
    for (auto it = object.begin(); it != object.end(); ++it) {
        bool known = false;
        for (const auto name : allowed) { known |= it.key() == name; }
        require(known,"unknown deployment field");
    }
}

uint32_t number(const json & value, uint32_t maximum) {
    require(value.is_number_unsigned() && value.get<uint64_t>() <= maximum,"invalid deployment integer");
    return value.get<uint32_t>();
}

std::string string(const json & value, size_t maximum) {
    require(value.is_string(),"invalid deployment string");
    const auto result = value.get<std::string>();
    require(!result.empty() && result.size() <= maximum,"deployment string length exceeds limit");
    for (const unsigned char byte : result) { require(byte >= 32 && byte != 127,"deployment string contains control bytes"); }
    return result;
}

std::string address(const json & value) {
    auto result = string(value,15);
    size_t start = 0;
    for (size_t part = 0; part < 4; ++part) {
        const auto dot = result.find('.',start);
        const size_t end = dot == std::string::npos ? result.size() : dot;
        require(end > start && end-start <= 3 && ((part == 3) == (dot == std::string::npos)) &&
                (end-start == 1 || result[start] != '0'),"deployment requires canonical numeric IPv4");
        unsigned byte = 0;
        const auto parsed = std::from_chars(result.data()+start,result.data()+end,byte);
        require(parsed.ec == std::errc() && parsed.ptr == result.data()+end && byte <= 255 &&
                (part || (byte > 0 && byte < 224 && byte != 127)),"deployment requires three explicit non-loopback unicast interfaces");
        start = end+1;
    }
    return result;
}

std::filesystem::path local_path(const json & value, const std::filesystem::path & directory) {
    const auto text = string(value,32768);
    require(text.find("://") == std::string::npos && text.rfind("//",0) != 0 && text.rfind("\\\\",0) != 0,
            "deployment paths must be local file paths");
    auto result = std::filesystem::u8path(text);
    require(result.has_filename(),"deployment path has no filename");
    if (result.is_relative()) { result = directory/result; }
    result = result.lexically_normal();
    require(result.has_filename() && result.filename() != "." && result.filename() != "..","deployment path has no filename");
    return result;
}

digest intent(const deployment_config & config) {
    constexpr uint8_t domain[] = "LlamaGLM deployment intent v1";
    std::vector<uint8_t> encoded(domain,domain+sizeof(domain));
    const auto integer = [&](uint32_t value) { for (size_t i = 0; i < 4; ++i) { encoded.push_back(uint8_t(value>>(8*i))); } };
    const auto text = [&](const std::string & value) {
        integer(uint32_t(value.size())); encoded.insert(encoded.end(),value.begin(),value.end());
    };
    integer(1); text("unsloth/GLM-5.3-Flash-GGUF"); text("a38483c8cd5df544f53d70fb281afe97369d5ab6"); text("UD-IQ3_XXS");
    integer(config.bounds.context); integer(config.bounds.rows); integer(config.bounds.frame_bytes);
    for (auto end : config.layer_ends) { integer(end); }
    for (const auto & node : config.nodes) {
        integer(node.rank); text(node.address); integer(node.data_port); integer(node.control_port); integer(node.api_port);
    }
    return sha256({{encoded.data(),encoded.size()}});
}

} // namespace

deployment_config parse_deployment(const std::string & text, const std::filesystem::path & directory) {
    require(!text.empty() && text.size() <= max_deployment_bytes,"deployment configuration exceeds byte limit");
    require(directory.is_absolute(),"deployment base directory must be absolute");
    json value;
    try {
        std::vector<std::set<std::string>> objects;
        const auto callback = [&](int depth, json::parse_event_t event, json & parsed) {
            require(depth <= 16,"deployment JSON nesting exceeds limit");
            if (event == json::parse_event_t::object_start) { objects.emplace_back(); }
            if (event == json::parse_event_t::key) {
                require(!objects.empty() && objects.back().insert(parsed.get<std::string>()).second,"duplicate deployment JSON field");
            }
            if (event == json::parse_event_t::object_end) { objects.pop_back(); }
            return true;
        };
        value = json::parse(text,callback);
    } catch (...) { throw std::invalid_argument("invalid or duplicate deployment JSON"); }
    try {
        fields(value,{"version","rank","model","nodes","secret_file","context","row_limit","frame_bytes","layer_ends"},5);
        require(number(value.at("version"),1) == 1,"unknown deployment version");
        deployment_config result; result.rank = number(value.at("rank"),2);
        if (value.contains("context")) { result.bounds.context = number(value.at("context"),1048576); }
        if (value.contains("row_limit")) { result.bounds.rows = number(value.at("row_limit"),2048); }
        if (value.contains("frame_bytes")) { result.bounds.frame_bytes = number(value.at("frame_bytes"),max_frame_bytes); }
        result.bounds.validate();
        if (value.contains("layer_ends")) {
            const auto & ends = value.at("layer_ends"); require(ends.is_array() && ends.size() == 3,"invalid deployment split");
            for (size_t i = 0; i < 3; ++i) { result.layer_ends[i] = number(ends[i],45); }
        }
        require(result.layer_ends[0] && result.layer_ends[0] < result.layer_ends[1] &&
                result.layer_ends[1] < result.layer_ends[2] && result.layer_ends[2] == 45,"deployment must own all 45 target layers exactly once");
        const auto & model = value.at("model"); fields(model,{"repository","revision","quant","files"},4);
        require(model.at("repository") == "unsloth/GLM-5.3-Flash-GGUF" &&
                model.at("revision") == "a38483c8cd5df544f53d70fb281afe97369d5ab6" && model.at("quant") == "UD-IQ3_XXS",
                "deployment requires the exact pinned dynamic mixed-quant target");
        const auto & files = model.at("files"); require(files.is_array() && files.size() == 4,"deployment requires all four local model files");
        for (size_t i = 0; i < 4; ++i) {
            result.model_files[i] = local_path(files[i],directory);
            require(result.model_files[i].filename().u8string() == "GLM-5.3-Flash-UD-IQ3_XXS-0000"+std::to_string(i+1)+"-of-00004.gguf",
                    "deployment model filenames/order differ from the pinned four shards");
        }
        result.secret_file = local_path(value.at("secret_file"),directory);
        for (const auto & file : result.model_files) { require(file != result.secret_file,"deployment credential and model paths overlap"); }
        const auto & nodes = value.at("nodes"); require(nodes.is_array() && nodes.size() == 3,"deployment requires all three owners");
        std::array<bool,3> seen{}; std::set<std::string> addresses;
        for (const auto & item : nodes) {
            fields(item,{"rank","address","data_port","control_port","api_port"},5);
            const auto rank = number(item.at("rank"),2); require(!seen[rank],"duplicate deployment owner"); seen[rank] = true;
            auto & node = result.nodes[rank]; node.rank = rank; node.address = address(item.at("address"));
            require(addresses.insert(node.address).second,"deployment owners require three distinct interfaces");
            node.data_port = uint16_t(number(item.at("data_port"),65535));
            node.control_port = uint16_t(number(item.at("control_port"),65535));
            node.api_port = uint16_t(number(item.at("api_port"),65535));
            require(rank == 0 ? (!node.data_port && !node.control_port && node.api_port) :
                    (node.data_port && node.control_port && node.data_port != node.control_port && !node.api_port),
                    "deployment ports violate coordinator/data/control ownership");
        }
        result.intent_digest = intent(result);
        return result;
    } catch (const json::exception &) { throw std::invalid_argument("missing or incorrectly typed deployment field"); }
}

deployment_config load_deployment(const std::filesystem::path & path) {
    std::ifstream input(path,std::ios::binary); require(bool(input),"cannot open local deployment configuration");
    std::string text(max_deployment_bytes+1,'\0'); input.read(text.data(),std::streamsize(text.size()));
    require(!input.bad(),"cannot read local deployment configuration"); text.resize(size_t(input.gcount()));
    return parse_deployment(text,std::filesystem::absolute(path).parent_path());
}

} // namespace glm_rpc
