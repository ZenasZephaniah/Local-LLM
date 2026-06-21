#pragma once
//
// telemetry_ui.hpp
// The full interactive TUI, shared by every entry point (standalone demo, live
// inference, replay). Panels:
//   1. Model Topology       - collapsible tree, j/k navigate, Space selects the
//                             capture target, l/Enter expand, h collapse
//   2. Live Packet Stream   - real-time table draining the SPSC ring
//   3. Attention Matrix     - grayscale heatmap of the selected layer; h/j/k/l
//                             pan, +/- contrast, f fullscreen
//   4. Runtime Metrics      - shape, dtype, sparsity gauge, latency delta
//   5. Anomaly Ledger       - timestamped log of flagged packets
// Tab cycles focus; q/Esc quits. The event-loop thread is the single ring
// consumer (it drains inside the event handler).
//
#include "layer_packet.hpp"
#include "packet_ring.hpp"
#include "capture_target.hpp"
#include "attention_slot.hpp"
#include "trace_io.hpp"

#include <ftxui/component/component.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>

#include <atomic>
#include <thread>
#include <vector>
#include <deque>
#include <unordered_map>
#include <string>
#include <chrono>
#include <algorithm>
#include <cstdio>
#include <ctime>

namespace llmtrace {
namespace ui {

using namespace ftxui;

// ---- small string helpers -------------------------------------------------
inline const char* kind_str(LayerKind k) {
    switch (k) {
        case LayerKind::Embedding: return "Embed";
        case LayerKind::Attention: return "Attn";
        case LayerKind::Mlp:       return "MLP";
        case LayerKind::Norm:      return "Norm";
        case LayerKind::Output:    return "Out";
        case LayerKind::Rope:      return "RoPE";
        default:                   return "Other";
    }
}
inline const char* dtype_str(DType d) {
    switch (d) { case DType::F32: return "float32"; case DType::F16: return "float16";
                 case DType::BF16: return "bfloat16"; case DType::Q4: return "Q4_K";
                 case DType::Q8: return "Q8_0"; default: return "other"; }
}
inline const char* dev_str(Device d) {
    switch (d) { case Device::Cuda: return "CUDA"; case Device::Metal: return "Metal";
                 case Device::Cpu: return "CPU"; default: return "?"; }
}

// ---- topology tree --------------------------------------------------------
struct TNode {
    std::string full, label;
    int         depth = 0;
    LayerKind   kind  = LayerKind::Other;
    int         layer_index = -1;
    bool        expandable = false, expanded = false, capturable = false;
    int         parent = -1;
};

class TopologyModel {
public:
    TopologyModel(int n_layers, const std::string& model_name) {
        add({model_name, model_name, 0, LayerKind::Other, -1, false, false, false, -1});
        add({"token_embd", "token_embd", 0, LayerKind::Embedding, -1, false, false, true, -1});
        for (int L = 0; L < n_layers; ++L) {
            const int blk = (int)nodes_.size();
            add({"blk." + std::to_string(L), "blk." + std::to_string(L),
                 0, LayerKind::Other, L, true, L == 0, false, -1});
            add({"blk." + std::to_string(L) + ".attn_norm", "attn_norm", 1, LayerKind::Norm,      L, false, false, true, blk});
            add({"blk." + std::to_string(L) + ".attn",      "attn",      1, LayerKind::Attention, L, false, false, true, blk});
            add({"blk." + std::to_string(L) + ".ffn_norm",  "ffn_norm",  1, LayerKind::Norm,      L, false, false, true, blk});
            add({"blk." + std::to_string(L) + ".ffn",       "ffn",       1, LayerKind::Mlp,       L, false, false, true, blk});
        }
        add({"output_norm", "output_norm", 0, LayerKind::Norm,   -1, false, false, true, -1});
        add({"output",      "output",      0, LayerKind::Output, -1, false, false, true, -1});
        cursor_ = 1;
    }

    std::vector<int> visible() const {
        std::vector<int> v;
        for (int i = 0; i < (int)nodes_.size(); ++i)
            if (nodes_[i].depth == 0 || nodes_[nodes_[i].parent].expanded) v.push_back(i);
        return v;
    }
    void move(int d) {
        const auto v = visible();
        int pos = 0;
        for (int i = 0; i < (int)v.size(); ++i) if (v[i] == cursor_) { pos = i; break; }
        cursor_ = v[std::max(0, std::min((int)v.size() - 1, pos + d))];
    }
    void toggle()  { if (nodes_[cursor_].expandable) nodes_[cursor_].expanded = !nodes_[cursor_].expanded; }
    void collapse(){ TNode& n = nodes_[cursor_];
                     if (n.expandable && n.expanded) n.expanded = false;
                     else if (n.parent >= 0) cursor_ = n.parent; }
    std::string select(CaptureTarget& t) {
        const TNode& n = nodes_[cursor_];
        if (!n.capturable) return sel_;
        selected_ = cursor_; t.set(n.layer_index, n.kind); sel_ = n.full;
        return sel_;
    }
    const std::vector<TNode>& nodes() const { return nodes_; }
    int cursor()   const { return cursor_; }
    int selected() const { return selected_; }

private:
    void add(TNode n) { nodes_.push_back(std::move(n)); }
    std::vector<TNode> nodes_;
    int cursor_ = 0, selected_ = -1;
    std::string sel_ = "(none)";
};

// ---- consumed-packet store ------------------------------------------------
class TelemetryStore {
public:
    struct LayerStat { LayerPacket last{}; double avg_latency_us = 0.0; std::uint64_t count = 0; };
    struct AnomalyEntry { std::string time, msg; std::uint8_t flags = 0; };

    void ingest(PacketRing& ring, TraceWriter* recorder) {
        LayerPacket p{};
        while (ring.try_pop(p)) {
            if (recorder) recorder->write(p);  // record exactly what is consumed
            if (first_ts_ == 0) first_ts_ = p.timestamp_ns;
            recent_.push_back(p);
            if (recent_.size() > kMaxRows) recent_.pop_front();
            ++total_;

            LayerStat& s = stats_[key_of(p.layer_index, p.kind)];
            ++s.count;
            s.avg_latency_us += (p.latency_us - s.avg_latency_us) / (double)s.count;
            s.last = p;
            latest_ = p; has_latest_ = true;

            if (p.anomaly) {
                anomalies_.push_back({wall_now(), describe(p), p.anomaly});
                if (anomalies_.size() > kMaxAnom) anomalies_.pop_front();
            }
        }
    }

    const std::deque<LayerPacket>&  recent()    const { return recent_; }
    const std::deque<AnomalyEntry>& anomalies() const { return anomalies_; }
    std::uint64_t total()    const { return total_; }
    std::uint64_t first_ts() const { return first_ts_; }
    const LayerStat* stat(std::int32_t layer, LayerKind k) const {
        auto it = stats_.find(key_of(layer, k));
        return it == stats_.end() ? nullptr : &it->second;
    }
    bool latest(LayerPacket& out) const { if (!has_latest_) return false; out = latest_; return true; }

private:
    static std::uint32_t key_of(std::int32_t layer, LayerKind k) {
        return (std::uint32_t)((layer + 1) << 8) | (std::uint8_t)k;
    }
    static std::string describe(const LayerPacket& p) {
        std::string m = p.name; char tail[48];
        if      (p.anomaly & AnomalyFlag::NonFinite)        m += "  NaN/Inf in activations";
        else if (p.anomaly & AnomalyFlag::ActivationBlowup) { std::snprintf(tail, sizeof(tail), "  outlier max=%.1f", p.max_abs); m += tail; }
        else if (p.anomaly & AnomalyFlag::SparsityCollapse) m += "  sparsity collapse (mean~0)";
        else if (p.anomaly & AnomalyFlag::DeviceFallback)   m += "  CPU fallback";
        return m;
    }
    static std::string wall_now() {
        using namespace std::chrono;
        auto now = system_clock::now();
        std::time_t t = system_clock::to_time_t(now);
        auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;
        std::tm tmv{};
#if defined(_WIN32)
        localtime_s(&tmv, &t);
#else
        localtime_r(&t, &tmv);
#endif
        char b[16];
        std::snprintf(b, sizeof(b), "%02d:%02d:%02d.%03d",
                      tmv.tm_hour, tmv.tm_min, tmv.tm_sec, (int)ms.count());
        return b;
    }

    static constexpr std::size_t kMaxRows = 500;
    static constexpr std::size_t kMaxAnom = 200;
    std::deque<LayerPacket>  recent_;
    std::deque<AnomalyEntry> anomalies_;
    std::unordered_map<std::uint32_t, LayerStat> stats_;
    LayerPacket   latest_{};
    bool          has_latest_ = false;
    std::uint64_t total_ = 0, first_ts_ = 0;
};

// ---- attention viewport state ---------------------------------------------
struct AttnView { int row = 0, col = 0; float contrast = 1.0f; bool full = false; };

// ---- render functions -----------------------------------------------------
inline Element render_topology(const TopologyModel& topo, bool focused) {
    Elements rows;
    for (int idx : topo.visible()) {
        const TNode& n = topo.nodes()[idx];
        std::string glyph = n.expandable ? (n.expanded ? "v " : "> ")
                                         : (n.capturable ? "* " : "  ");
        Element e = text(std::string(n.depth * 2, ' ') + glyph + n.label);
        if (idx == topo.selected()) e = e | color(Color::GreenLight) | bold;
        if (idx == topo.cursor())   e = e | inverted | focus;
        rows.push_back(e);
    }
    std::string title = focused ? " 1. MODEL TOPOLOGY  [focus] " : " 1. MODEL TOPOLOGY ";
    return window(text(title) | (focused ? bold : dim),
                  vbox(std::move(rows)) | vscroll_indicator | yframe | flex);
}

inline Element render_stream(const TelemetryStore& store, bool focused) {
    Elements rows;
    rows.push_back(text("   ID   TIME(ms)  LAYER             TYPE   DEV    LAT(ms)") | bold | dim);
    rows.push_back(separator());
    for (const LayerPacket& p : store.recent()) {
        const double rel = store.first_ts() ? (double)(p.timestamp_ns - store.first_ts()) / 1e6 : 0.0;
        char buf[112];
        std::snprintf(buf, sizeof(buf), "%5llu %9.1f  %-17s %-6s %-6s %6.2f",
                      (unsigned long long)p.id, rel, p.name, kind_str(p.kind),
                      dev_str(p.device), p.latency_us / 1000.0);
        Element e = text(buf);
        if (p.anomaly & AnomalyFlag::DeviceFallback) e = e | color(Color::RGB(255, 140, 0));
        else if (p.anomaly)                          e = e | color(Color::RedLight);
        rows.push_back(e);
    }
    if (store.recent().size()) rows.back() = rows.back() | focus;
    std::string title = focused ? " 2. LIVE PACKET STREAM  [focus] " : " 2. LIVE PACKET STREAM ";
    return window(text(title) | (focused ? bold : dim),
                  vbox(std::move(rows)) | vscroll_indicator | yframe | flex);
}

inline Element render_attention(const AttnSlot& slot, const AttnView& view, bool focused) {
    std::string title = focused ? " 3. ATTENTION MATRIX  [focus] " : " 3. ATTENTION MATRIX ";
    AttentionSnapshot s;
    if (!slot.load(s) || s.rows == 0) {
        return window(text(title) | (focused ? bold : dim), vbox({
            text("  no attention captured yet") | dim,
            text("  select an 'attn' node with Space; run with flash-attn disabled") | dim,
        }));
    }
    const int vh = std::min(s.rows, view.full ? 28 : 8);
    const int vw = std::min(s.cols, view.full ? 48 : 8);
    const int r0 = std::clamp(view.row, 0, std::max(0, s.rows - vh));
    const int c0 = std::clamp(view.col, 0, std::max(0, s.cols - vw));

    Elements grid;
    for (int r = 0; r < vh; ++r) {
        Elements cells;
        for (int c = 0; c < vw; ++c) {
            float w = s.weights[(std::size_t)(r0 + r) * kAttnTile + (c0 + c)] * view.contrast;
            w = std::clamp(w, 0.0f, 1.0f);
            int g = (int)(w * 255.0f);
            cells.push_back(text("  ") | bgcolor(Color::RGB(g / 3, g / 2, g))); // brand-blue ramp
        }
        grid.push_back(hbox(std::move(cells)));
    }
    char info[128];
    std::snprintf(info, sizeof(info),
                  "layer %d  head %d   rows [%d-%d] cols [%d-%d]   contrast x%.2f%s",
                  s.layer_index, s.head, r0, r0 + vh - 1, c0, c0 + vw - 1, view.contrast,
                  view.full ? "   [fullscreen]" : "");
    return window(text(title) | (focused ? bold : dim),
                  vbox({ text(info) | dim, separator(), vbox(std::move(grid)) }));
}

inline Element render_metrics(const TelemetryStore& store, const CaptureTarget& target) {
    const std::int32_t tl = target.layer_index.load(std::memory_order_relaxed);
    const LayerKind     tk = (LayerKind)target.kind.load(std::memory_order_acquire);
    const TelemetryStore::LayerStat* s = store.stat(tl, tk);

    LayerPacket p{}; double delta = 0.0; bool have = false;
    if (s)                    { p = s->last; delta = p.latency_us - s->avg_latency_us; have = true; }
    else if (store.latest(p)) { have = true; }

    if (!have)
        return window(text(" 4. RUNTIME METRICS ") | dim, text("  waiting for packets...") | dim);

    std::string shp = "[";
    for (std::uint8_t i = 0; i < p.ndim; ++i) { shp += std::to_string((long long)p.shape[i]); if (i + 1 < p.ndim) shp += ", "; }
    shp += "]";
    char meanmax[64]; std::snprintf(meanmax, sizeof(meanmax), "% .3f / %.3f", p.mean, p.max_abs);
    char latbuf[80];  std::snprintf(latbuf, sizeof(latbuf), "%.3f ms   (delta % .3f ms)", p.latency_us / 1000.0, delta / 1000.0);
    char pct[16];     std::snprintf(pct, sizeof(pct), "%.1f%%", p.sparsity * 100.0f);

    Color lat_color = (s && p.latency_us > 1.5 * s->avg_latency_us) ? Color::Yellow : Color::GreenLight;
    Color sp_color  = p.sparsity > 0.8f ? Color::Yellow : Color::Cyan;
    auto field = [](const char* k, Element v) { return hbox({ text(k) | dim, std::move(v) }); };

    return window(text(" 4. RUNTIME METRICS ") | dim, vbox({
        field("Node          : ", text(p.name) | bold),
        field("Tensor Shape  : ", text(shp)),
        field("Dtype         : ", text(dtype_str(p.dtype))),
        field("Mean / Max|x| : ", text(meanmax)),
        field("Latency       : ", text(latbuf) | color(lat_color)),
        hbox({ text("Sparsity      : ") | dim, text(pct) }),
        gauge(p.sparsity) | color(sp_color),
    }));
}

inline Element render_ledger(const TelemetryStore& store, bool focused) {
    std::string title = focused ? " 5. ANOMALY LEDGER  [focus] " : " 5. ANOMALY LEDGER ";
    Elements rows;
    for (const auto& a : store.anomalies()) {
        const char* icon = (a.flags & AnomalyFlag::NonFinite)        ? "x "
                         : (a.flags & AnomalyFlag::ActivationBlowup) ? "! " : "~ ";
        Color c = (a.flags & AnomalyFlag::NonFinite)        ? Color::RedLight
                : (a.flags & AnomalyFlag::ActivationBlowup) ? Color::Yellow : Color::Magenta;
        rows.push_back(text(a.time + "  " + icon + a.msg) | color(c));
    }
    if (rows.empty()) rows.push_back(text("  no anomalies") | dim);
    else              rows.back() = rows.back() | focus;
    return window(text(title) | (focused ? bold : dim),
                  vbox(std::move(rows)) | vscroll_indicator | yframe | flex);
}

// ---- entry point ----------------------------------------------------------
struct TuiConfig {
    PacketRing&        ring;
    CaptureTarget&     target;
    AttnSlot&          attn;
    std::atomic<bool>& stop;       // set true by run_tui on quit; producers watch it
    std::string        model_name;
    int                n_layers = 0;
    std::string        mode = "LIVE";
    TraceWriter*       recorder = nullptr;
};

inline int run_tui(const TuiConfig& cfg) {
    auto screen = ScreenInteractive::Fullscreen();

    TopologyModel  topo(cfg.n_layers, cfg.model_name);
    TelemetryStore store;
    AttnView       aview;
    int            focus_panel = 0;
    constexpr int  kPanels = 4; // topology, stream, attention, ledger
    std::string    selection = "(none)";

    auto root = Renderer([&] {
        const std::int32_t tl = cfg.target.layer_index.load(std::memory_order_relaxed);
        std::string tgt = (tl < 0 && selection == "(none)") ? "(none)" : selection;

        Element top_bar = text(" LLM-TRACE  [Tab] focus  [j/k] nav  [Space] select  "
                               "[h/j/k/l] pan  [+/-] contrast  [f] fullscreen  [q] quit ") | inverted;
        Element status = hbox({
            text(" mode: ") | dim, text(cfg.mode) | color(Color::Cyan) | bold,
            text("   target: ") | dim, text(tgt) | color(Color::GreenLight) | bold,
            filler(),
            text("packets: " + std::to_string(store.total())) | dim,
            text("   dropped: " + std::to_string(cfg.ring.dropped())) | dim, text(" "),
        });

        Element body;
        if (aview.full) {
            body = render_attention(cfg.attn, aview, focus_panel == 2) | flex;
        } else {
            Element top_row = hbox({
                render_topology(topo, focus_panel == 0) | size(WIDTH, EQUAL, 38),
                render_stream(store, focus_panel == 1) | flex,
            }) | flex;
            Element mid_row = render_attention(cfg.attn, aview, focus_panel == 2)
                              | size(HEIGHT, EQUAL, 13);
            Element bot_row = hbox({
                render_metrics(store, cfg.target) | size(WIDTH, EQUAL, 46),
                render_ledger(store, focus_panel == 3) | flex,
            }) | size(HEIGHT, EQUAL, 11);
            body = vbox({ top_row, mid_row, bot_row }) | flex;
        }
        return vbox({ top_bar, body, status });
    });

    root |= CatchEvent([&](Event e) -> bool {
        store.ingest(cfg.ring, cfg.recorder); // single consumer = UI thread
        if (e == Event::Custom) return false;

        if (e == Event::Character('q') || e == Event::Escape) {
            cfg.stop.store(true, std::memory_order_release);
            screen.Exit();
            return true;
        }
        if (e == Event::Tab) { focus_panel = (focus_panel + 1) % kPanels; return true; }

        if (focus_panel == 0) { // topology
            if (e == Event::Character('j') || e == Event::ArrowDown)  { topo.move(1);  return true; }
            if (e == Event::Character('k') || e == Event::ArrowUp)    { topo.move(-1); return true; }
            if (e == Event::Character('l') || e == Event::Return || e == Event::ArrowRight) { topo.toggle();   return true; }
            if (e == Event::Character('h') || e == Event::ArrowLeft)  { topo.collapse(); return true; }
            if (e == Event::Character(' '))                           { selection = topo.select(cfg.target); return true; }
        } else if (focus_panel == 2) { // attention viewport
            if (e == Event::Character('j') || e == Event::ArrowDown)  { aview.row += 1; return true; }
            if (e == Event::Character('k') || e == Event::ArrowUp)    { aview.row = std::max(0, aview.row - 1); return true; }
            if (e == Event::Character('l') || e == Event::ArrowRight) { aview.col += 1; return true; }
            if (e == Event::Character('h') || e == Event::ArrowLeft)  { aview.col = std::max(0, aview.col - 1); return true; }
            if (e == Event::Character('+') || e == Event::Character('=')) { aview.contrast = std::min(8.0f, aview.contrast + 0.25f); return true; }
            if (e == Event::Character('-') || e == Event::Character('_')) { aview.contrast = std::max(0.25f, aview.contrast - 0.25f); return true; }
            if (e == Event::Character('f'))                           { aview.full = !aview.full; return true; }
        }
        return false;
    });

    std::thread refresher([&] {
        while (!cfg.stop.load(std::memory_order_acquire)) {
            screen.PostEvent(Event::Custom);
            std::this_thread::sleep_for(std::chrono::milliseconds(33));
        }
    });

    screen.Loop(root);
    cfg.stop.store(true, std::memory_order_release);
    refresher.join();
    return 0;
}

} // namespace ui
} // namespace llmtrace
