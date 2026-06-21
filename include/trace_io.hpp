#pragma once
//
// trace_io.hpp
// Record the LayerPacket stream to a flat binary file and replay it later with
// no model loaded. Because LayerPacket is trivially copyable, the on-disk format
// is just a small header followed by raw packed records.
//
#include <cstdio>
#include <cstring>
#include <cstdint>
#include "layer_packet.hpp"

namespace llmtrace {

struct TraceHeader {
    char          magic[8];     // "LLMTRC1"
    std::uint32_t version;      // format version
    std::int32_t  n_layers;     // transformer blocks, for the topology tree
    char          model[48];    // model name for display
};

inline void fill_header(TraceHeader& h, std::int32_t n_layers, const char* model) {
    std::memset(&h, 0, sizeof(h));
    std::memcpy(h.magic, "LLMTRC1", 7);
    h.version  = 1;
    h.n_layers = n_layers;
    std::strncpy(h.model, model ? model : "model", sizeof(h.model) - 1);
}

class TraceWriter {
public:
    TraceWriter(const char* path, std::int32_t n_layers, const char* model) {
        f_ = std::fopen(path, "wb");
        if (f_) {
            TraceHeader h{};
            fill_header(h, n_layers, model);
            std::fwrite(&h, sizeof(h), 1, f_);
        }
    }
    ~TraceWriter() { if (f_) std::fclose(f_); }

    bool ok() const { return f_ != nullptr; }
    void write(const LayerPacket& p) { if (f_) std::fwrite(&p, sizeof(p), 1, f_); }

    TraceWriter(const TraceWriter&) = delete;
    TraceWriter& operator=(const TraceWriter&) = delete;

private:
    std::FILE* f_ = nullptr;
};

class TraceReader {
public:
    explicit TraceReader(const char* path) {
        f_ = std::fopen(path, "rb");
        if (f_ && std::fread(&header_, sizeof(header_), 1, f_) == 1 &&
            std::memcmp(header_.magic, "LLMTRC1", 7) == 0) {
            valid_ = true;
        }
    }
    ~TraceReader() { if (f_) std::fclose(f_); }

    bool ok() const { return valid_; }
    std::int32_t n_layers() const { return header_.n_layers; }
    const char*  model()    const { return header_.model; }

    bool next(LayerPacket& out) {
        if (!valid_) return false;
        return std::fread(&out, sizeof(out), 1, f_) == 1;
    }

    TraceReader(const TraceReader&) = delete;
    TraceReader& operator=(const TraceReader&) = delete;

private:
    std::FILE*  f_ = nullptr;
    TraceHeader header_{};
    bool        valid_ = false;
};

} // namespace llmtrace
