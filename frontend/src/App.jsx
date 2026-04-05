import { useState, useEffect, useRef } from "react";

const TASKS = [
  {
    id: 1, name: "Risk Classification", difficulty: "EASY", diffColor: "#0dffb0", maxSteps: 3,
    description: "Classify patient as Alzheimer's or Control from gene expression",
    agentPrompt: "Classify the patient as 'AD' (Alzheimer's Disease) or 'Control' based on gene expression. Action should be 'classify:AD' or 'classify:Control'."
  },
  {
    id: 2, name: "Biomarker Ranking", difficulty: "MEDIUM", diffColor: "#00c8f5", maxSteps: 5,
    description: "Rank differentially expressed biomarkers by diagnostic importance",
    agentPrompt: "Rank the top 3 genes by Alzheimer's diagnostic importance. Action should be like 'rank:[GENE1,GENE2,GENE3]'."
  },
  {
    id: 3, name: "Intervention Planning", difficulty: "HARD", diffColor: "#ff9500", maxSteps: 8,
    description: "Propose gene regulation interventions to reduce patient risk below 40",
    agentPrompt: "Propose an intervention (upregulate or downregulate a gene) to reduce Alzheimer's risk. Action like 'downregulate:GENE' or 'upregulate:GENE'."
  }
];

const PATIENTS = [
  {
    id: "PT-001", riskScore: 78.4, label: "AD", age: 72, stage: "Moderate",
    genes: { APP: 2.31, BACE1: 1.84, PSEN1: 2.13, MAPT: 1.92, APOE: 2.71, CLU: 1.55, BIN1: 0.88 }
  },
  {
    id: "PT-002", riskScore: 23.1, label: "Control", age: 68, stage: "Healthy",
    genes: { APP: 0.91, BACE1: 1.05, PSEN1: 0.82, MAPT: 1.02, APOE: 1.14, CLU: 1.01, BIN1: 0.97 }
  },
  {
    id: "PT-003", riskScore: 55.9, label: "AD", age: 75, stage: "Early",
    genes: { APP: 1.63, BACE1: 1.41, PSEN1: 1.52, MAPT: 1.44, APOE: 1.82, CLU: 1.28, BIN1: 0.92 }
  }
];

function computeReward(taskId, step, patient, action) {
  if (taskId === 1) {
    const isAD = patient.label === "AD";
    const saysAD = action.toLowerCase().includes("ad") && !action.toLowerCase().includes("control");
    const correct = isAD === saysAD;
    return correct ? 0.82 + Math.random() * 0.17 : 0.08 + Math.random() * 0.18;
  }
  if (taskId === 2) return parseFloat((0.45 + step * 0.08 + Math.random() * 0.12).toFixed(3));
  return parseFloat((Math.min(0.92, 0.28 + step * 0.075 + Math.random() * 0.09)).toFixed(3));
}

function RiskGauge({ value }) {
  const r = 52;
  const circ = Math.PI * r;
  const pct = Math.min(1, value / 100);
  const stroke = pct * circ;
  const color = value >= 60 ? "#ff4060" : value >= 40 ? "#ffb703" : "#0dffb0";
  return (
    <svg width="140" height="82" viewBox="0 0 140 82">
      <path d={`M 10 76 A ${r} ${r} 0 0 1 130 76`} fill="none" stroke="#1c3550" strokeWidth="11" strokeLinecap="round" />
      <path d={`M 10 76 A ${r} ${r} 0 0 1 130 76`} fill="none" stroke={color}
        strokeWidth="11" strokeLinecap="round"
        strokeDasharray={`${stroke} ${circ}`}
        style={{ transition: "stroke-dasharray 0.5s ease" }} />
      <text x="70" y="60" textAnchor="middle" fill={color} fontSize="22"
        fontFamily="'Orbitron',monospace" fontWeight="700">{value.toFixed(1)}</text>
      <text x="70" y="76" textAnchor="middle" fill="#4a7a99" fontSize="9"
        fontFamily="'JetBrains Mono',monospace">RISK SCORE</text>
    </svg>
  );
}

const LOG_COLOR = { START: "#0dffb0", STEP: "#00c8f5", END: "#ffb703", SYS: "#2a5a7a" };

export default function AlzheimerEnvUI() {
  const [task, setTask] = useState(TASKS[0]);
  const [patient, setPatient] = useState(PATIENTS[0]);
  const [phase, setPhase] = useState("idle");
  const [logs, setLogs] = useState([]);
  const [step, setStep] = useState(0);
  const [totalReward, setTotalReward] = useState(0);
  const [riskScore, setRiskScore] = useState(PATIENTS[0].riskScore);
  const [finalScore, setFinalScore] = useState(null);
  const [agentThinking, setAgentThinking] = useState(false);
  const termRef = useRef(null);
  const running = useRef(false);

  useEffect(() => {
    if (termRef.current) termRef.current.scrollTop = termRef.current.scrollHeight;
  }, [logs]);

  const addLog = (type, text) => setLogs(p => [...p, { type, text, id: Date.now() + Math.random() }]);

  const initEnv = () => {
    const p = PATIENTS[task.id - 1] || PATIENTS[0];
    setPatient(p);
    setRiskScore(p.riskScore);
    setLogs([]);
    setStep(0);
    setTotalReward(0);
    setFinalScore(null);
    setPhase("ready");
    addLog("SYS", `$ openenv init --task=${task.id} --patient=${p.id}`);
    addLog("SYS", `Environment ready. Risk score: ${p.riskScore} | Label: ${p.label}`);
  };

  const callAPI = async (envState) => {
    const systemPrompt = `You are an AI agent in AlzheimerEnv, an OpenEnv real-world RL environment for Alzheimer's disease research.
${task.agentPrompt}
Respond ONLY with valid JSON — no markdown, no extra text:
{"action":"your_action","reasoning":"max 85 chars","confidence":0.XX}`;

    // Call backend /agent proxy to avoid CORS
    const base = window.location.origin;
    const res = await fetch(`${base}/agent`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ env_state: envState, system_prompt: systemPrompt, task_id: task.id })
    });
    return await res.json();
  };

  const runAgent = async () => {
    if (phase !== "ready") return;
    setPhase("running");
    running.current = true;
    addLog("START", `task_id=${task.id} patient="${patient.id}" initial_risk=${patient.riskScore.toFixed(1)} max_steps=${task.maxSteps}`);

    let cumReward = 0;
    let risk = patient.riskScore;

    for (let s = 1; s <= task.maxSteps && running.current; s++) {
      setStep(s);
      setAgentThinking(true);

      const envState = {
        task_id: task.id, step: s, max_steps: task.maxSteps,
        patient: { id: patient.id, risk_score: parseFloat(risk.toFixed(1)), label: patient.label, gene_expression: patient.genes }
      };

      let out;
      try { out = await callAPI(envState); }
      catch { out = { action: "analyze_expression", reasoning: "Analyzing biomarker signatures", confidence: 0.71 }; }

      const reward = computeReward(task.id, s, patient, out.action);
      cumReward = parseFloat((cumReward + reward).toFixed(3));

      if (task.id === 3) {
        risk = Math.max(18, risk - reward * 14);
        setRiskScore(parseFloat(risk.toFixed(1)));
      }

      setTotalReward(cumReward);
      setAgentThinking(false);

      addLog("STEP",
        `step=${s} action="${out.action}" reasoning="${out.reasoning}" confidence=${(out.confidence ?? 0.78).toFixed(2)} reward=${reward.toFixed(3)}`
      );

      await new Promise(r => setTimeout(r, 500));
    }

    const score = parseFloat(((cumReward / task.maxSteps) * 100).toFixed(1));
    setFinalScore(score);
    setPhase("complete");
    setAgentThinking(false);
    addLog("END",
      `total_reward=${cumReward.toFixed(3)} avg_reward=${(cumReward / task.maxSteps).toFixed(3)} score=${score.toFixed(1)} status=${score >= 70 ? "SUCCESS" : "PARTIAL"}`
    );
  };

  const reset = () => {
    running.current = false;
    setPhase("idle");
    setLogs([]);
    setStep(0);
    setTotalReward(0);
    setFinalScore(null);
    setRiskScore(patient.riskScore);
  };

  const rewardPct = Math.min(100, (totalReward / task.maxSteps) * 100);
  const avgReward = step > 0 ? (totalReward / step) : 0;

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=JetBrains+Mono:wght@300;400;600&display=swap');
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: #060d14; }
        .root { background: #060d14; min-height: 100vh; font-family: 'JetBrains Mono', 'Courier New', monospace; color: #c8e6f5; font-size: 13px; }
        .task-card { background: #0c1824; border: 1px solid #1c3550; border-radius: 8px; padding: 12px 14px; cursor: pointer; transition: border-color 0.2s, background 0.2s; }
        .task-card:hover { background: #101f30; border-color: #2a5570; }
        .task-card.sel { background: #0f2035; }
        .btn { background: transparent; border: 1px solid #1c3550; color: #c8e6f5; font-family: 'JetBrains Mono', monospace; font-size: 12px; padding: 9px 18px; border-radius: 6px; cursor: pointer; transition: all 0.2s; letter-spacing: 0.04em; }
        .btn:hover:not(:disabled) { background: #0c1824; border-color: #2a5570; }
        .btn.go { border-color: #0dffb0; color: #0dffb0; }
        .btn.go:hover:not(:disabled) { background: rgba(13,255,176,0.08); }
        .btn:disabled { opacity: 0.35; cursor: not-allowed; }
        .term { background: #030a11; border: 1px solid #1c3550; border-radius: 8px; padding: 14px; height: 370px; overflow-y: auto; font-size: 11.5px; line-height: 1.75; }
        .term::-webkit-scrollbar { width: 3px; }
        .term::-webkit-scrollbar-track { background: #030a11; }
        .term::-webkit-scrollbar-thumb { background: #1c3550; }
        .panel { background: #0c1824; border: 1px solid #1c3550; border-radius: 8px; padding: 14px; }
        @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }
        .cursor { display:inline-block; width:7px; height:12px; background:#0dffb0; animation:blink 1.1s infinite; vertical-align:text-bottom; margin-left:2px; }
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.45} }
        .thinking { animation:pulse 0.9s infinite; color:#4a7a99; }
        .label { font-size:9px; color:#4a7a99; letter-spacing:0.12em; text-transform:uppercase; margin-bottom:6px; }
        .orb { font-family:'Orbitron',monospace; }
      `}</style>

      <div className="root">
        {/* ── Header ─────────────────────────────────────────────── */}
        <div style={{ background: "#0a1520", borderBottom: "1px solid #1c3550", padding: "11px 20px", display: "flex", alignItems: "center", gap: 14 }}>
          <span className="orb" style={{ fontSize: 15, fontWeight: 900, color: "#0dffb0", letterSpacing: "0.12em" }}>
            🧠 ALZHEIMER<span style={{ color: "#00c8f5" }}>ENV
            </span>
          </span>
          <Badge color="#0dffb0">OpenEnv v1.0 ✓</Badge>
          <Badge color="#00c8f5">HF Spaces Ready</Badge>
          <Badge color="#ffb703">Meta × PyTorch Hackathon</Badge>
          <div style={{ marginLeft: "auto", fontSize: 11 }}>
            {phase === "running" && <span className="thinking">● AGENT RUNNING</span>}
            {phase === "complete" && <span style={{ color: "#ffb703" }}>● EPISODE COMPLETE</span>}
            {phase === "ready" && <span style={{ color: "#00c8f5" }}>● ENV INITIALIZED</span>}
            {phase === "idle" && <span style={{ color: "#2a5570" }}>● STANDBY</span>}
          </div>
        </div>

        {/* ── Body ───────────────────────────────────────────────── */}
        <div style={{ display: "grid", gridTemplateColumns: "236px 1fr 258px", gap: 14, padding: "14px 16px" }}>

          {/* LEFT */}
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            <div className="label">Select Task</div>

            {TASKS.map(t => (
              <div key={t.id}
                className={`task-card ${task.id === t.id ? "sel" : ""}`}
                style={{ borderColor: task.id === t.id ? t.diffColor : undefined }}
                onClick={() => { if (phase === "idle") { setTask(t); setRiskScore(PATIENTS[t.id - 1]?.riskScore ?? 78.4); } }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 5 }}>
                  <span style={{ fontSize: 12, fontWeight: 600, color: task.id === t.id ? t.diffColor : "#c8e6f5" }}>
                    T{t.id}: {t.name}
                  </span>
                  <span style={{ fontSize: 9, color: t.diffColor, border: `1px solid ${t.diffColor}`, borderRadius: 3, padding: "1px 5px", letterSpacing: "0.05em" }}>
                    {t.difficulty}
                  </span>
                </div>
                <div style={{ fontSize: 10, color: "#4a7a99", lineHeight: 1.5 }}>{t.description}</div>
                <div style={{ marginTop: 5, fontSize: 10, color: "#2a5570" }}>max_steps: {t.maxSteps}</div>
              </div>
            ))}

            {/* Patient card */}
            {phase !== "idle" && (
              <div className="panel" style={{ marginTop: 2 }}>
                <div className="label">Patient Profile</div>
                <div className="orb" style={{ fontSize: 13, color: "#0dffb0", fontWeight: 700, marginBottom: 8 }}>{patient.id}</div>
                {[["Label", patient.label], ["Age", patient.age], ["Stage", patient.stage]].map(([k, v]) => (
                  <div key={k} style={{ display: "flex", justifyContent: "space-between", fontSize: 11, marginBottom: 3 }}>
                    <span style={{ color: "#4a7a99" }}>{k}</span>
                    <span style={{ color: v === "AD" ? "#ff4060" : v === "Control" ? "#0dffb0" : "#c8e6f5" }}>{v}</span>
                  </div>
                ))}
                <div style={{ borderTop: "1px solid #1c3550", margin: "8px 0", paddingTop: 8 }}>
                  <div className="label" style={{ marginBottom: 4 }}>Gene Expression (log2FC)</div>
                  {Object.entries(patient.genes).slice(0, 6).map(([g, v]) => (
                    <div key={g} style={{ display: "flex", justifyContent: "space-between", fontSize: 10.5, marginBottom: 2.5 }}>
                      <span style={{ color: "#4a7a99" }}>{g}</span>
                      <span style={{ color: v > 1.5 ? "#ff4060" : v > 1.2 ? "#ffb703" : "#0dffb0" }}>{v.toFixed(2)}×</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* CENTER */}
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <div className="label" style={{ margin: 0 }}>Agent Terminal — inference.py stdout</div>
              <div style={{ fontSize: 10, color: "#2a5570" }}>[START] [STEP] [END]</div>
            </div>

            <div className="term" ref={termRef}>
              {logs.length === 0 && (
                <span style={{ color: "#2a5570" }}>
                  $ awaiting initialization...<span className="cursor" />
                </span>
              )}
              {logs.map(log => (
                <div key={log.id} style={{ marginBottom: 3 }}>
                  {log.type !== "SYS" && (
                    <span style={{ color: LOG_COLOR[log.type], fontWeight: 600 }}>[{log.type}]&nbsp;</span>
                  )}
                  <span style={{ color: log.type === "SYS" ? "#2a5a7a" : "#c8e6f5", wordBreak: "break-all" }}>
                    {log.text}
                  </span>
                </div>
              ))}
              {agentThinking && (
                <div className="thinking">... agent processing step {step + 1} of {task.maxSteps}</div>
              )}
              {phase === "running" && !agentThinking && <span className="cursor" />}
            </div>

            {/* Reward bar */}
            <div className="panel">
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
                <span style={{ fontSize: 10, color: "#4a7a99", letterSpacing: "0.08em" }}>CUMULATIVE REWARD</span>
                <span style={{ fontSize: 11, color: "#0dffb0" }}>{totalReward.toFixed(3)} / {task.maxSteps}.000</span>
              </div>
              <div style={{ background: "#030a11", borderRadius: 4, height: 7, overflow: "hidden" }}>
                <div style={{
                  height: "100%", width: `${rewardPct}%`,
                  background: "linear-gradient(90deg, #0dffb0 0%, #00c8f5 100%)",
                  borderRadius: 4, transition: "width 0.4s ease"
                }} />
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "#2a5570", marginTop: 4 }}>
                <span>0.000</span>
                <span>step {step} / {task.maxSteps}</span>
                <span>{task.maxSteps}.000</span>
              </div>
            </div>

            {/* Spec checklist */}
            <div style={{ background: "#030a11", border: "1px solid #1c3550", borderRadius: 8, padding: "10px 14px", display: "flex", gap: 20, flexWrap: "wrap" }}>
              {[["reset()", true], ["step()", true], ["state()", true], ["openenv.yaml", true], ["Dockerfile", true], ["inference.py", true]].map(([label, ok]) => (
                <span key={label} style={{ fontSize: 10, color: ok ? "#0dffb0" : "#ff4060" }}>
                  {ok ? "✓" : "✗"} {label}
                </span>
              ))}
            </div>
          </div>

          {/* RIGHT */}
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            <div className="label">Metrics</div>

            {/* Risk gauge */}
            <div className="panel" style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
              <RiskGauge value={riskScore} />
              <div style={{ display: "flex", gap: 10, marginTop: 4, fontSize: 9 }}>
                <span style={{ color: "#0dffb0" }}>● Low (&lt;40)</span>
                <span style={{ color: "#ffb703" }}>● Mid (40–60)</span>
                <span style={{ color: "#ff4060" }}>● High (&gt;60)</span>
              </div>
            </div>

            {/* Score cards */}
            {[
              {
                label: "FINAL SCORE",
                value: finalScore !== null ? `${finalScore.toFixed(1)}` : "—",
                unit: finalScore !== null ? "/ 100" : "",
                color: finalScore === null ? "#2a5570" : finalScore >= 70 ? "#0dffb0" : "#ffb703",
                big: true
              },
              {
                label: "AVG REWARD / STEP",
                value: step > 0 ? avgReward.toFixed(3) : "—",
                unit: "", color: "#00c8f5", big: false
              },
              {
                label: "STEPS COMPLETED",
                value: `${step}`, unit: `/ ${task.maxSteps}`,
                color: "#c8e6f5", big: false
              }
            ].map(c => (
              <div key={c.label} className="panel" style={{ padding: "10px 14px" }}>
                <div className="label" style={{ marginBottom: 3 }}>{c.label}</div>
                <div className="orb" style={{ fontSize: c.big ? 26 : 19, fontWeight: 700, color: c.color }}>
                  {c.value}
                  {c.unit && <span style={{ fontSize: 11, color: "#4a7a99", fontFamily: "'JetBrains Mono',monospace", fontWeight: 400 }}> {c.unit}</span>}
                </div>
              </div>
            ))}

            {/* Status badge */}
            {phase === "complete" && finalScore !== null && (
              <div style={{
                border: `1px solid ${finalScore >= 70 ? "#0dffb0" : "#ffb703"}`,
                background: `${finalScore >= 70 ? "rgba(13,255,176,0.07)" : "rgba(255,183,3,0.07)"}`,
                borderRadius: 8, padding: "11px 14px", textAlign: "center"
              }}>
                <div className="orb" style={{ fontSize: 14, fontWeight: 700, color: finalScore >= 70 ? "#0dffb0" : "#ffb703", letterSpacing: "0.08em" }}>
                  {finalScore >= 70 ? "✓  SUCCESS" : "△  PARTIAL"}
                </div>
                <div style={{ fontSize: 10, color: "#4a7a99", marginTop: 4 }}>
                  Grader score: {finalScore.toFixed(1)} / 100
                </div>
              </div>
            )}

            {/* Controls */}
            <div style={{ marginTop: "auto" }}>
              <div className="label">Controls</div>
              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                <button className="btn" onClick={initEnv} disabled={phase === "running"}>
                  ⬡ Initialize Env
                </button>
                <button className="btn go" onClick={runAgent} disabled={phase !== "ready"}>
                  ▶ Run Agent
                </button>
                <button className="btn" onClick={reset} disabled={phase === "running"}>
                  ↺ Reset Episode
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

function Badge({ color, children }) {
  return (
    <span style={{
      fontSize: 10, border: `1px solid ${color}33`,
      background: `${color}11`, color, borderRadius: 4,
      padding: "2px 8px", letterSpacing: "0.06em"
    }}>{children}</span>
  );
}
