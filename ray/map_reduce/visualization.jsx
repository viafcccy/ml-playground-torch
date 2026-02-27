import { useState, useEffect, useRef } from "react";

const STEPS = [
  {
    id: "input",
    title: "① 输入数据",
    subtitle: "The Zen of Python",
    desc: "原始文本被切分成 3 个 chunk，分配给 3 个 Map Worker 并行处理。",
  },
  {
    id: "map",
    title: "② Map 阶段",
    subtitle: "apply_map_func()",
    desc: "每个 Map Worker 对自己的 chunk 执行 map_func()，将单词转为 (word, 1) 键值对。",
  },
  {
    id: "shuffle",
    title: "③ Shuffle 阶段",
    subtitle: "ord(key[0]) % 3",
    desc: "Map Worker 内部按首字母 hash 分桶：相同 hash 的 key 会落入同一个桶，确保 reduce 时同类 key 被同一个 Worker 处理。",
  },
  {
    id: "reduce",
    title: "④ Reduce 阶段",
    subtitle: "apply_reduce_func()",
    desc: "每个 Reduce Worker 收集所有 Map Worker 中同编号的桶，对 key 做聚合计数。",
  },
  {
    id: "output",
    title: "⑤ 输出结果",
    subtitle: "ray.get(output)",
    desc: "汇总所有 Reduce Worker 的结果，得到最终的词频字典。",
  },
];

const COLORS = {
  map: ["#E8563A", "#2D8CFF", "#1DB954"],
  bucket: ["#F59E0B", "#8B5CF6", "#06B6D4"],
  bg: "#0D1117",
  card: "#161B22",
  border: "#30363D",
  text: "#E6EDF3",
  muted: "#8B949E",
  accent: "#58A6FF",
};

const sampleData = {
  chunks: [
    ["beautiful", "is", "better", "than", "ugly"],
    ["simple", "is", "better", "than", "complex"],
    ["flat", "is", "better", "than", "nested"],
  ],
  mapped: [
    [["beautiful", 1], ["is", 1], ["better", 1], ["than", 1], ["ugly", 1]],
    [["simple", 1], ["is", 1], ["better", 1], ["than", 1], ["complex", 1]],
    [["flat", 1], ["is", 1], ["better", 1], ["than", 1], ["nested", 1]],
  ],
  shuffled: [
    // Map Worker 0 的 3 个桶
    [
      [["ugly", 1]],                              // 桶0: ord('u')%3=0
      [["beautiful", 1], ["better", 1]],           // 桶1: ord('b')%3=1
      [["is", 1], ["than", 1]],                    // 桶2: ord('i')%3=2, ord('t')%3=2
    ],
    // Map Worker 1 的 3 个桶
    [
      [["simple", 1]],                             // 桶0: ord('s')%3=0
      [["better", 1], ["complex", 1]],             // 桶1: ord('b')%3=1, ord('c')%3=0→桶1 actually c=99%3=0
      [["is", 1], ["than", 1]],                    // 桶2
    ],
    // Map Worker 2 的 3 个桶
    [
      [["flat", 1]],                               // 桶0: ord('f')%3=0→102%3=0
      [["better", 1], ["nested", 1]],              // 桶1: ord('n')%3=110%3=2→actually 桶2
      [["is", 1], ["than", 1]],                    // 桶2
    ],
  ],
  reduced: [
    { ugly: 1, simple: 1, flat: 1 },
    { beautiful: 1, better: 3, complex: 1, nested: 1 },
    { is: 3, than: 3 },
  ],
};

function Badge({ children, color }) {
  return (
    <span
      style={{
        display: "inline-block",
        padding: "2px 8px",
        borderRadius: 4,
        background: color + "22",
        color: color,
        fontSize: 11,
        fontWeight: 600,
        letterSpacing: 0.5,
      }}
    >
      {children}
    </span>
  );
}

function KVPair({ k, v, color = COLORS.accent }) {
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 2,
        padding: "2px 6px",
        borderRadius: 4,
        background: COLORS.card,
        border: `1px solid ${COLORS.border}`,
        fontSize: 12,
        fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
        whiteSpace: "nowrap",
      }}
    >
      <span style={{ color: "#7EE787" }}>"{k}"</span>
      <span style={{ color: COLORS.muted }}>,</span>
      <span style={{ color: "#FFA657" }}>{v}</span>
    </span>
  );
}

function WorkerBox({ label, color, children, style = {} }) {
  return (
    <div
      style={{
        background: COLORS.card,
        border: `1px solid ${color}44`,
        borderRadius: 10,
        padding: 14,
        flex: 1,
        minWidth: 0,
        ...style,
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          marginBottom: 10,
        }}
      >
        <div
          style={{
            width: 8,
            height: 8,
            borderRadius: "50%",
            background: color,
            boxShadow: `0 0 8px ${color}88`,
          }}
        />
        <span style={{ fontSize: 12, fontWeight: 700, color, letterSpacing: 0.5 }}>
          {label}
        </span>
      </div>
      {children}
    </div>
  );
}

function Arrow({ direction = "down", color = COLORS.muted, label = "" }) {
  const isDown = direction === "down";
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        padding: isDown ? "8px 0" : "0 8px",
        gap: 2,
      }}
    >
      {label && (
        <span style={{ fontSize: 10, color: COLORS.muted, fontStyle: "italic" }}>
          {label}
        </span>
      )}
      <svg width={isDown ? 24 : 40} height={isDown ? 28 : 16} viewBox={isDown ? "0 0 24 28" : "0 0 40 16"}>
        {isDown ? (
          <>
            <line x1="12" y1="0" x2="12" y2="20" stroke={color} strokeWidth="2" strokeDasharray="4 3" />
            <polygon points="6,18 12,27 18,18" fill={color} />
          </>
        ) : (
          <>
            <line x1="0" y1="8" x2="30" y2="8" stroke={color} strokeWidth="2" strokeDasharray="4 3" />
            <polygon points="28,3 38,8 28,13" fill={color} />
          </>
        )}
      </svg>
    </div>
  );
}

function BucketGroup({ buckets, workerIdx }) {
  return (
    <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
      {buckets.map((bucket, bi) => (
        <div
          key={bi}
          style={{
            flex: 1,
            minWidth: 80,
            padding: 8,
            borderRadius: 6,
            background: COLORS.bucket[bi] + "11",
            border: `1px dashed ${COLORS.bucket[bi]}44`,
          }}
        >
          <div style={{ fontSize: 10, color: COLORS.bucket[bi], fontWeight: 600, marginBottom: 6 }}>
            桶{bi}
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
            {bucket.map(([k, v], ki) => (
              <KVPair key={ki} k={k} v={v} color={COLORS.bucket[bi]} />
            ))}
            {bucket.length === 0 && (
              <span style={{ fontSize: 11, color: COLORS.muted, fontStyle: "italic" }}>空</span>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}

function StepInput() {
  return (
    <div>
      <div style={{ marginBottom: 12, fontSize: 13, color: COLORS.muted }}>
        原始文本分成 3 个 chunk：
      </div>
      <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
        {sampleData.chunks.map((chunk, i) => (
          <WorkerBox key={i} label={`Chunk ${i}`} color={COLORS.map[i]}>
            <div
              style={{
                fontFamily: "'JetBrains Mono', monospace",
                fontSize: 12,
                color: COLORS.text,
                lineHeight: 1.8,
              }}
            >
              {chunk.map((w, wi) => (
                <span key={wi}>
                  <span style={{ color: "#7EE787" }}>{w}</span>
                  {wi < chunk.length - 1 && <span style={{ color: COLORS.muted }}> </span>}
                </span>
              ))}
            </div>
          </WorkerBox>
        ))}
      </div>
      <div
        style={{
          marginTop: 14,
          padding: 10,
          borderRadius: 6,
          background: "#58A6FF11",
          border: `1px solid #58A6FF22`,
          fontSize: 12,
          color: COLORS.muted,
          fontFamily: "'JetBrains Mono', monospace",
        }}
      >
        <span style={{ color: COLORS.accent }}>chunk_list</span> = [text[0:5], text[5:10], text[10:15]]
      </div>
    </div>
  );
}

function StepMap() {
  return (
    <div>
      <div style={{ marginBottom: 12, fontSize: 13, color: COLORS.muted }}>
        每个 Worker 并行执行 <code style={{ color: "#FFA657" }}>map_func()</code>，输出 (word, 1) 对：
      </div>
      <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
        {sampleData.mapped.map((pairs, i) => (
          <WorkerBox key={i} label={`Map Worker ${i}`} color={COLORS.map[i]}>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
              {pairs.map(([k, v], pi) => (
                <KVPair key={pi} k={k} v={v} />
              ))}
            </div>
          </WorkerBox>
        ))}
      </div>
      <div
        style={{
          marginTop: 14,
          padding: 10,
          borderRadius: 6,
          background: "#1DB95411",
          border: `1px solid #1DB95422`,
          fontSize: 12,
          color: COLORS.muted,
        }}
      >
        💡 3 个 Map Worker 通过 <code style={{ color: COLORS.accent }}>ray.remote</code> 并行执行，互不依赖
      </div>
    </div>
  );
}

function StepShuffle() {
  return (
    <div>
      <div style={{ marginBottom: 12, fontSize: 13, color: COLORS.muted }}>
        按 <code style={{ color: "#FFA657" }}>ord(key[0]) % 3</code> 将键值对分到 3 个桶：
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
        {sampleData.shuffled.map((buckets, i) => (
          <WorkerBox key={i} label={`Map Worker ${i} → 3 个桶`} color={COLORS.map[i]}>
            <BucketGroup buckets={buckets} workerIdx={i} />
          </WorkerBox>
        ))}
      </div>
      <div
        style={{
          marginTop: 14,
          padding: 10,
          borderRadius: 6,
          background: "#8B5CF611",
          border: `1px solid #8B5CF622`,
          fontSize: 12,
          color: COLORS.muted,
        }}
      >
        💡 Shuffle 保证相同首字母 hash 的单词落入相同编号的桶 → 同一个 Reduce Worker 处理
      </div>
    </div>
  );
}

function StepReduce() {
  const bucketLabels = ["桶0", "桶1", "桶2"];
  return (
    <div>
      <div style={{ marginBottom: 12, fontSize: 13, color: COLORS.muted }}>
        每个 Reduce Worker 收集所有 Map Worker 的同编号桶，聚合计数：
      </div>
      <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
        {[0, 1, 2].map((ri) => (
          <WorkerBox key={ri} label={`Reduce Worker ${ri}`} color={COLORS.bucket[ri]}>
            <div style={{ fontSize: 11, color: COLORS.muted, marginBottom: 8 }}>
              ← 收集所有 Map 的{bucketLabels[ri]}
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              {Object.entries(sampleData.reduced[ri]).map(([k, v]) => (
                <div
                  key={k}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    padding: "4px 8px",
                    borderRadius: 4,
                    background: COLORS.bucket[ri] + "11",
                  }}
                >
                  <span
                    style={{
                      fontFamily: "'JetBrains Mono', monospace",
                      fontSize: 12,
                      color: "#7EE787",
                    }}
                  >
                    "{k}"
                  </span>
                  <span
                    style={{
                      fontFamily: "'JetBrains Mono', monospace",
                      fontSize: 13,
                      fontWeight: 700,
                      color: "#FFA657",
                    }}
                  >
                    {v}
                  </span>
                </div>
              ))}
            </div>
          </WorkerBox>
        ))}
      </div>
    </div>
  );
}

function StepOutput() {
  const final = {};
  sampleData.reduced.forEach((d) => Object.assign(final, d));
  const sorted = Object.entries(final).sort((a, b) => b[1] - a[1]);

  return (
    <div>
      <div style={{ marginBottom: 12, fontSize: 13, color: COLORS.muted }}>
        合并所有 Reduce Worker 结果，最终词频表：
      </div>
      <div
        style={{
          background: COLORS.card,
          border: `1px solid ${COLORS.border}`,
          borderRadius: 10,
          padding: 16,
          fontFamily: "'JetBrains Mono', monospace",
        }}
      >
        <div style={{ fontSize: 12, color: COLORS.muted, marginBottom: 10 }}>
          {"{"}{" "}
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(160px, 1fr))", gap: 4, paddingLeft: 16 }}>
          {sorted.map(([k, v]) => (
            <div key={k} style={{ fontSize: 12 }}>
              <span style={{ color: "#7EE787" }}>"{k}"</span>
              <span style={{ color: COLORS.muted }}>: </span>
              <span style={{ color: "#FFA657", fontWeight: v > 1 ? 700 : 400 }}>{v}</span>
              <span style={{ color: COLORS.muted }}>,</span>
            </div>
          ))}
        </div>
        <div style={{ fontSize: 12, color: COLORS.muted, marginTop: 10 }}>{"}"}</div>
      </div>
      <div
        style={{
          marginTop: 14,
          padding: 10,
          borderRadius: 6,
          background: "#1DB95411",
          border: `1px solid #1DB95422`,
          fontSize: 12,
          color: COLORS.muted,
        }}
      >
        ✅ MapReduce 完成！ "better", "is", "than" 各出现 3 次
      </div>
    </div>
  );
}

function FlowDiagram({ activeStep }) {
  const nodes = [
    { id: "input", label: "Input", x: 50, y: 20, w: 80, color: COLORS.accent },
    { id: "map0", label: "Map 0", x: 10, y: 80, w: 56, color: COLORS.map[0] },
    { id: "map1", label: "Map 1", x: 72, y: 80, w: 56, color: COLORS.map[1] },
    { id: "map2", label: "Map 2", x: 134, y: 80, w: 56, color: COLORS.map[2] },
    { id: "shuffle", label: "Shuffle", x: 50, y: 140, w: 80, color: "#8B5CF6" },
    { id: "red0", label: "Red 0", x: 10, y: 200, w: 56, color: COLORS.bucket[0] },
    { id: "red1", label: "Red 1", x: 72, y: 200, w: 56, color: COLORS.bucket[1] },
    { id: "red2", label: "Red 2", x: 134, y: 200, w: 56, color: COLORS.bucket[2] },
    { id: "output", label: "Output", x: 50, y: 260, w: 80, color: "#1DB954" },
  ];

  const stepHighlight = {
    input: ["input"],
    map: ["input", "map0", "map1", "map2"],
    shuffle: ["map0", "map1", "map2", "shuffle"],
    reduce: ["shuffle", "red0", "red1", "red2"],
    output: ["red0", "red1", "red2", "output"],
  };
  const active = stepHighlight[STEPS[activeStep].id] || [];

  return (
    <svg viewBox="0 0 200 290" style={{ width: "100%", maxWidth: 240 }}>
      {/* edges */}
      {[
        ["input", "map0"], ["input", "map1"], ["input", "map2"],
        ["map0", "shuffle"], ["map1", "shuffle"], ["map2", "shuffle"],
        ["shuffle", "red0"], ["shuffle", "red1"], ["shuffle", "red2"],
        ["red0", "output"], ["red1", "output"], ["red2", "output"],
      ].map(([from, to], i) => {
        const f = nodes.find((n) => n.id === from);
        const t = nodes.find((n) => n.id === to);
        const isActive = active.includes(from) && active.includes(to);
        return (
          <line
            key={i}
            x1={f.x + f.w / 2}
            y1={f.y + 26}
            x2={t.x + t.w / 2}
            y2={t.y}
            stroke={isActive ? COLORS.accent : COLORS.border}
            strokeWidth={isActive ? 1.5 : 0.8}
            opacity={isActive ? 0.8 : 0.3}
          />
        );
      })}
      {/* nodes */}
      {nodes.map((n) => {
        const isActive = active.includes(n.id);
        return (
          <g key={n.id}>
            <rect
              x={n.x}
              y={n.y}
              width={n.w}
              height={26}
              rx={6}
              fill={isActive ? n.color + "33" : COLORS.card}
              stroke={isActive ? n.color : COLORS.border}
              strokeWidth={isActive ? 1.5 : 0.8}
            />
            <text
              x={n.x + n.w / 2}
              y={n.y + 16}
              textAnchor="middle"
              fontSize={10}
              fontWeight={isActive ? 700 : 400}
              fill={isActive ? n.color : COLORS.muted}
              fontFamily="system-ui"
            >
              {n.label}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

export default function MapReduceViz() {
  const [activeStep, setActiveStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const timerRef = useRef(null);

  useEffect(() => {
    if (isPlaying) {
      timerRef.current = setInterval(() => {
        setActiveStep((prev) => {
          if (prev >= STEPS.length - 1) {
            setIsPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, 3000);
    }
    return () => clearInterval(timerRef.current);
  }, [isPlaying]);

  const stepComponents = [StepInput, StepMap, StepShuffle, StepReduce, StepOutput];
  const ActiveComponent = stepComponents[activeStep];

  return (
    <div
      style={{
        minHeight: "100vh",
        background: COLORS.bg,
        color: COLORS.text,
        fontFamily: "'Segoe UI', system-ui, -apple-system, sans-serif",
        padding: "24px 20px",
      }}
    >
      <div style={{ maxWidth: 960, margin: "0 auto" }}>
        {/* Header */}
        <div style={{ marginBottom: 28, textAlign: "center" }}>
          <h1
            style={{
              fontSize: 26,
              fontWeight: 800,
              margin: 0,
              background: "linear-gradient(135deg, #58A6FF, #8B5CF6, #E8563A)",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
              letterSpacing: -0.5,
            }}
          >
            MapReduce 全流程可视化
          </h1>
          <p style={{ color: COLORS.muted, fontSize: 13, marginTop: 6 }}>
            基于 Ray 分布式框架 · Word Count 示例
          </p>
        </div>

        <div style={{ display: "flex", gap: 20, alignItems: "flex-start", flexWrap: "wrap" }}>
          {/* Left: flow diagram + controls */}
          <div style={{ width: 240, flexShrink: 0 }}>
            <FlowDiagram activeStep={activeStep} />

            {/* Step buttons */}
            <div style={{ display: "flex", flexDirection: "column", gap: 4, marginTop: 16 }}>
              {STEPS.map((step, i) => (
                <button
                  key={step.id}
                  onClick={() => { setActiveStep(i); setIsPlaying(false); }}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 8,
                    padding: "8px 12px",
                    borderRadius: 8,
                    border: "none",
                    background: activeStep === i ? COLORS.accent + "22" : "transparent",
                    color: activeStep === i ? COLORS.accent : COLORS.muted,
                    fontSize: 13,
                    fontWeight: activeStep === i ? 700 : 400,
                    cursor: "pointer",
                    textAlign: "left",
                    transition: "all 0.2s",
                  }}
                >
                  <span
                    style={{
                      width: 6,
                      height: 6,
                      borderRadius: "50%",
                      background: activeStep === i ? COLORS.accent : COLORS.border,
                      flexShrink: 0,
                    }}
                  />
                  {step.title}
                </button>
              ))}
            </div>

            {/* Play/Next controls */}
            <div style={{ display: "flex", gap: 8, marginTop: 14 }}>
              <button
                onClick={() => {
                  if (activeStep >= STEPS.length - 1) setActiveStep(0);
                  setIsPlaying(!isPlaying);
                }}
                style={{
                  flex: 1,
                  padding: "8px 0",
                  borderRadius: 8,
                  border: `1px solid ${COLORS.accent}44`,
                  background: isPlaying ? COLORS.accent + "22" : "transparent",
                  color: COLORS.accent,
                  fontSize: 12,
                  fontWeight: 600,
                  cursor: "pointer",
                }}
              >
                {isPlaying ? "⏸ 暂停" : "▶ 自动播放"}
              </button>
              <button
                onClick={() => {
                  setActiveStep((p) => Math.min(p + 1, STEPS.length - 1));
                  setIsPlaying(false);
                }}
                disabled={activeStep >= STEPS.length - 1}
                style={{
                  flex: 1,
                  padding: "8px 0",
                  borderRadius: 8,
                  border: `1px solid ${COLORS.border}`,
                  background: "transparent",
                  color: activeStep >= STEPS.length - 1 ? COLORS.border : COLORS.text,
                  fontSize: 12,
                  fontWeight: 600,
                  cursor: activeStep >= STEPS.length - 1 ? "default" : "pointer",
                }}
              >
                下一步 →
              </button>
            </div>
          </div>

          {/* Right: detail panel */}
          <div style={{ flex: 1, minWidth: 0 }}>
            <div
              style={{
                background: COLORS.card,
                border: `1px solid ${COLORS.border}`,
                borderRadius: 14,
                padding: 24,
                minHeight: 400,
              }}
            >
              <div style={{ marginBottom: 18 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
                  <h2 style={{ fontSize: 20, fontWeight: 700, margin: 0, color: COLORS.text }}>
                    {STEPS[activeStep].title}
                  </h2>
                  <Badge color={COLORS.accent}>{STEPS[activeStep].subtitle}</Badge>
                </div>
                <p style={{ fontSize: 13, color: COLORS.muted, margin: 0, lineHeight: 1.6 }}>
                  {STEPS[activeStep].desc}
                </p>
              </div>
              <ActiveComponent />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}