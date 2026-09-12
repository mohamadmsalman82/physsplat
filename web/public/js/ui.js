/**
 * The interface.
 *
 * Built against the `app` object main.js exposes, and nothing else, so the
 * page's markup can change without touching the physics. Everything here is
 * plain DOM: the page has no build step and needs none.
 *
 * Layout: a top bar with the scene picker (photo thumbnails) and the engine
 * badge; a right-hand dock with Play, Camera, Physics, Display and Sensors
 * tabs; a transport bar at the bottom (reset, play/pause, step, slow motion);
 * a card for the selected pencil at bottom left; labels over pencils when
 * asked for; a help sheet on "?"; H hides the lot.
 */

const $ = (id) => document.getElementById(id);
const el = (tag, attrs = {}, ...children) => {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") n.className = v;
    else if (k === "html") n.innerHTML = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else if (v !== null && v !== undefined) n.setAttribute(k, v);
  }
  for (const c of children) if (c !== null && c !== undefined) n.append(c);
  return n;
};
const fmt = (x, d = 2) => (Number.isFinite(x) ? x.toFixed(d) : "–");

const ICON = {
  play: '<svg viewBox="0 0 16 16"><path d="M4 2.5v11l9-5.5z"/></svg>',
  pause: '<svg viewBox="0 0 16 16"><path d="M3.5 2.5h3v11h-3zM9.5 2.5h3v11h-3z"/></svg>',
  step: '<svg viewBox="0 0 16 16"><path d="M3 3v10l6-5zM10.5 3h2v10h-2z"/></svg>',
  reset: '<svg viewBox="0 0 16 16"><path d="M8 3a5 5 0 1 1-4.9 6h1.6A3.5 3.5 0 1 0 8 4.5V7L4 4l4-3z"/></svg>',
  plus: '<svg viewBox="0 0 16 16"><path d="M7 2h2v5h5v2H9v5H7V9H2V7h5z"/></svg>',
};

function slider({ label, min, max, step, value, unit = "", format = (v) => fmt(v, 2), onInput }) {
  const out = el("b", {}, format(value) + unit);
  const input = el("input", { type: "range", min, max, step, value });
  const paint = () => { input.style.setProperty("--p", `${((input.value - min) / (max - min)) * 100}%`); };
  input.addEventListener("input", () => { const v = +input.value; out.textContent = format(v) + unit; paint(); onInput(v); });
  paint();
  const wrap = el("div", { class: "slider" }, el("div", { class: "top" }, el("span", {}, label), out), input);
  wrap.set = (v) => { input.value = v; out.textContent = format(v) + unit; paint(); };
  return wrap;
}
function toggle({ label, value, onChange, hint = null }) {
  const sw = el("div", { class: "switch" + (value ? " on" : ""), role: "switch" });
  const row = el("div", { class: "row" }, el("label", {}, label), sw);
  sw.addEventListener("click", () => { sw.classList.toggle("on"); onChange(sw.classList.contains("on")); });
  row.set = (v) => sw.classList.toggle("on", !!v);
  return hint ? el("div", {}, row, el("div", { class: "hint" }, hint)) : row;
}
function segmented(options, value, onChange) {
  const seg = el("div", { class: "seg" });
  const buttons = options.map(([val, text]) => {
    const b = el("button", { onclick: () => { select(val); onChange(val); } }, text);
    b.dataset.val = String(val); return b;
  });
  const select = (v) => buttons.forEach((b) => b.classList.toggle("active", b.dataset.val === String(v)));
  buttons.forEach((b) => seg.append(b));
  select(value);
  seg.set = select;
  return seg;
}
const group = (title, ...children) => el("div", { class: "group" }, el("h3", {}, title), ...children);
const btn = (text, onclick, cls = "") => el("button", { class: "btn " + cls, onclick }, text);

export function initUI(app) {
  const root = $("ui");
  let toastTimer = null;
  const toast = (msg, ms = 1800) => {
    const t = $("toast"); t.textContent = msg; t.hidden = false;
    clearTimeout(toastTimer); toastTimer = setTimeout(() => { t.hidden = true; }, ms);
  };
  app.toast = toast;

  // ------------------------------------------------------------ top bar
  const scenes = $("scenes");
  const paintScenes = () => {
    scenes.replaceChildren(...app.scenes.map((name) => el("button",
      { class: "scene" + (name === app.scene ? " active" : ""), title: name,
        onclick: () => { app.loadScene(name); } },
      el("img", { src: `./desk/${name}_thumb.jpg`, alt: name, onerror: (e) => { e.target.remove(); } }),
      el("small", {}, name.replace("IMG_", "#")))));
  };
  paintScenes();
  app.onScene(paintScenes);

  const engine = $("engine");
  const paintEngine = () => {
    engine.classList.toggle("gnn", app.engine === "gnn");
    engine.querySelector("b").textContent = app.engine === "gnn" ? "learned GNN" : "Rapier";
  };
  paintEngine();
  engine.addEventListener("click", () => app.switchEngine(app.engine === "gnn" ? "rapier" : "gnn"));

  $("helpbtn").addEventListener("click", () => { $("help").hidden = !$("help").hidden; });
  $("helpclose").addEventListener("click", () => { $("help").hidden = true; });
  $("hidebtn").addEventListener("click", () => root.classList.toggle("hidden"));

  // ------------------------------------------------------------- dock
  const dock = $("dock"), tabs = $("tabs"), panels = $("panels");
  $("dockfold").addEventListener("click", () => {
    dock.classList.toggle("collapsed");
    $("dockfold").textContent = dock.classList.contains("collapsed") ? "‹" : "›";
  });
  const panelOf = {};
  const addTab = (name) => {
    const t = el("div", { class: "tab", onclick: () => showTab(name) }, name);
    const p = el("div", { class: "panel" });
    tabs.append(t); panels.append(p); panelOf[name] = { t, p };
    return p;
  };
  const showTab = (name) => {
    for (const [n, { t, p }] of Object.entries(panelOf)) { t.classList.toggle("active", n === name); p.classList.toggle("active", n === name); }
    activeTab = name;
    // toggles reflect the real state, which keys and the console can change too
    if (name === "Display") for (const [k, row] of Object.entries(displayRows)) row.set?.(app.display.get(k));
  };
  let activeTab = "Play";

  // ---- Play
  {
    const p = addTab("Play");
    const speed = segmented([[0.1, "0.1×"], [0.25, "¼×"], [0.5, "½×"], [1, "1×"]], app.speed, (v) => app.setSpeed(v));
    app.onSpeed((v) => speed.set(v));
    p.append(
      group("Playback",
        el("div", { class: "row" }, el("label", {}, "Speed"), speed),
        el("div", { class: "hint" }, "Slow motion is real physics at a smaller step rate, not a replay."),
        el("div", { class: "btns" },
          btn("Reset scene", () => app.reset()),
          btn("Drop a pencil", () => { const i = app.spawn(); toast(`pencil ${i} dropped`); }, "primary"),
          btn("Lift one", () => { const b = app.selected ?? 0; app.probe("lift", b, { height: 0.05 }); toast(`lifting pencil ${b}`); }),
          btn("Pull bottom", () => { app.probe("pullBottom"); toast("pulling the load-bearing pencil"); }))),
      group("Interaction",
        slider({ label: "Grab strength", min: 10, max: 60, step: 1, value: app.physics.get("grabStrength"), unit: " ω",
          format: (v) => v.toFixed(0), onInput: (v) => app.physics.set("grabStrength", v) }),
        slider({ label: "Follow speed", min: 0.1, max: 1.0, step: 0.05, value: app.physics.get("followSpeed"), unit: " m/s",
          onInput: (v) => app.physics.set("followSpeed", v) }),
        el("div", { class: "hint" }, "The grab is a spring between your cursor and the point you pressed. Follow speed caps how fast that point chases the cursor.")),
    );
  }

  // ---- Camera
  {
    const p = addTab("Camera");
    const presets = el("div", { class: "btns" },
      ...[["home", "Home ⁠1"], ["top", "Top 2"], ["side", "Side 3"], ["low", "Low 4"], ["photo", "Photo 5"]]
        .map(([k, t]) => btn(t, () => app.camera.preset(k), "small")));
    const followRow = toggle({ label: "Follow selected pencil", value: app.camera.following !== null,
      onChange: (on) => app.camera.follow(on ? (app.selected ?? 0) : null) });
    const orbitRow = toggle({ label: "Auto-orbit", value: false, onChange: (on) => app.camera.autoOrbit(on) });
    const orbitSpeed = slider({ label: "Orbit speed", min: 0.05, max: 1.0, step: 0.05, value: 0.25, unit: " rad/s",
      onInput: (v) => app.camera.autoOrbit(true, v) });
    const fov = slider({ label: "Field of view", min: 24, max: 80, step: 1, value: app.camera.state().fov, unit: "°",
      format: (v) => v.toFixed(0), onInput: (v) => app.camera.setFov(v) });
    const readout = el("div", { class: "kv" });
    app.camera.onChange((s) => { followRow.set(s.following !== null); orbitRow.set(s.orbiting); });
    p.append(
      group("Views", presets, el("div", { class: "hint" }, "Photo is where the picture was taken from, recovered by fitting a camera to the pencils; the desk under them is that photograph.")),
      group("Motion", followRow, orbitRow, orbitSpeed, fov),
      group("Now", readout));
    setInterval(() => {
      if (activeTab !== "Camera" || root.classList.contains("hidden")) return;
      const s = app.camera.state();
      readout.replaceChildren(
        el("span", {}, "elevation"), el("span", {}, `${fmt(s.elevation_deg, 1)}°`),
        el("span", {}, "azimuth"), el("span", {}, `${fmt(s.azimuth_deg, 1)}°`),
        el("span", {}, "distance"), el("span", {}, `${fmt(s.distance_m * 100, 1)} cm`),
        el("span", {}, "following"), el("span", {}, s.following === null ? "–" : `pencil ${s.following}`));
    }, 250);
  }

  // ---- Physics
  {
    const p = addTab("Physics");
    const live = app.physics.available;
    const g = slider({ label: "Gravity", min: 0, max: 2.5, step: 0.05, value: app.physics.get("gravity"), unit: " g",
      onInput: (v) => app.physics.set("gravity", v) });
    const fp = slider({ label: "Friction, pencil on pencil", min: 0, max: 1, step: 0.01, value: app.physics.get("frictionPencil"),
      onInput: (v) => app.physics.set("frictionPencil", v) });
    const fd = slider({ label: "Friction, pencil on desk", min: 0, max: 1, step: 0.01, value: app.physics.get("frictionDesk"),
      onInput: (v) => app.physics.set("frictionDesk", v) });
    const re = slider({ label: "Bounciness", min: 0, max: 0.8, step: 0.01, value: app.physics.get("restitution"),
      onInput: (v) => app.physics.set("restitution", v) });
    p.append(
      group("World",
        g, fp, fd, re,
        el("div", { class: "btns" }, btn("Real-world defaults", () => {
          app.physics.set("gravity", 1); app.physics.set("frictionPencil", 0.28);
          app.physics.set("frictionDesk", 0.45); app.physics.set("restitution", 0.04);
          g.set(1); fp.set(0.28); fd.set(0.45); re.set(0.04); toast("physics reset to measured defaults");
        }, "small")),
        el("div", { class: "hint" }, live
          ? "Live: every change takes effect on the next step. The defaults are the values that matched real pencils on the sensor checks."
          : "The learned engine has no adjustable contact parameters; switch to Rapier to change these.")),
      group("Pencil", el("div", { class: "kv" },
        el("span", {}, "object"), el("span", {}, "BIC Matic Grip"),
        el("span", {}, "length"), el("span", {}, "150.0 mm"),
        el("span", {}, "barrel / grip"), el("span", {}, "9.0 / 11.0 mm"),
        el("span", {}, "mass"), el("span", {}, "6.2 g"),
        el("span", {}, "collider"), el("span", {}, "5 convex hulls + clip"))));
    if (!live) for (const s of [g, fp, fd, re]) s.querySelector("input").disabled = true;
  }

  // ---- Display
  const displayRows = {};
  {
    const p = addTab("Display");
    const t = (key, label, hint) => {
      const row = toggle({ label, value: app.display.get(key), onChange: (v) => app.display.set(key, v), hint });
      displayRows[key] = row; return row;
    };
    p.append(
      group("Scene",
        t("desk", "Desk from the photograph"),
        t("comparePhoto", "Compare with the real photo", "Puts the original photograph, pencils and all, under the simulated ones."),
        t("shadows", "Shadows"),
        t("filmic", "Filmic tone mapping")),
      group("Overlays",
        t("labels", "Pencil labels"),
        t("contacts", "Contact points", "Where pencils touch each other and the desk, from the sensor layer."),
        t("velocities", "Velocity arrows"),
        t("colliders", "Collider outlines", "The convex hulls the solver actually collides.")));
  }

  // ---- Sensors
  {
    const p = addTab("Sensors");
    const pre = el("pre", { class: "mono", style: "white-space:pre-wrap;margin:0;color:var(--muted);max-height:46vh;overflow:auto;font-size:11px;line-height:1.4" }, "…");
    const events = el("div", { class: "mono", style: "font-size:11px;color:var(--warn);margin-top:8px" });
    p.append(
      group("Live report",
        el("div", { class: "btns", style: "margin-bottom:8px" },
          btn("Copy report", async () => { try { await navigator.clipboard.writeText(app.sensors.report()); toast("report copied"); } catch { toast("copy failed"); } }, "small"),
          btn("Diagnostics panel (D)", () => dispatchEvent(new KeyboardEvent("keydown", { key: "d" })), "small")),
        pre, events),
      el("div", { class: "hint" }, "Every reading is derived from the simulator's own state and the pencil's known shape: which part of which pencil touches which part of which other, in millimetres from the point."));
    setInterval(() => {
      if (activeTab !== "Sensors" || root.classList.contains("hidden") || !app.sensors) return;
      try {
        pre.textContent = app.sensors.report();
        const ev = app.sensors.events({ n: 600 }).filter((e) => !/moving with nothing/.test(e.kind)).slice(0, 4);
        events.textContent = ev.length ? ev.map((e) => `${e.kind}: ${e.bodies ? "bodies " + e.bodies.join("+") : "pencil " + e.body}, steps ${e.from}–${e.to ?? e.from}`).join("\n") : "";
      } catch (e) { pre.textContent = String(e); }
    }, 400);
  }
  showTab("Play");

  // -------------------------------------------------------- transport
  {
    const bar = $("transport");
    const playBtn = el("button", { class: "big", title: "pause / play (Space)", html: ICON.pause, onclick: () => (app.paused ? app.play() : app.pause()) });
    const stepBtn = el("button", { class: "iconbtn", title: "step one frame (.)", html: ICON.step, onclick: () => app.step() });
    const resetBtn = el("button", { class: "iconbtn", title: "reset scene", html: ICON.reset, onclick: () => app.reset() });
    const spawnBtn = el("button", { class: "iconbtn", title: "drop a pencil (N)", html: ICON.plus, onclick: () => { const i = app.spawn(); toast(`pencil ${i} dropped`); } });
    const stats = el("span", { id: "stats" }, "loading…");
    bar.append(resetBtn, playBtn, stepBtn, spawnBtn, stats);
    app.onPause((paused) => { playBtn.innerHTML = paused ? ICON.play : ICON.pause; });
    setInterval(() => {
      const s = app.stats;
      if (!s) return;
      const rate = s.rate >= 0.98 ? "<b>real time</b>" : `${fmt(s.rate, 2)}× real time`;
      stats.innerHTML = `${s.engine} · ${fmt(s.stepMs, 1)} ms/step · ${rate} · ${s.steps} steps` +
        (app.speed < 1 ? ` · <b>${app.speed}× slow</b>` : "") + (app.paused ? " · <b>paused</b>" : "");
    }, 400);
  }

  // ------------------------------------------------------- selection card
  const card = $("card");
  const paintCard = () => {
    const i = app.selected;
    if (i === null || i === undefined) { card.hidden = true; return; }
    const b = app.bodyInfo(i);
    if (!b) { card.hidden = true; return; }
    card.hidden = false;
    const rest = b.resting_on.length ? b.resting_on.map((r) => (r === "table" ? "the desk" : `pencil ${r}`)).join(", ") : (b.unsupported ? "nothing (in the air)" : "—");
    card.replaceChildren(
      el("h4", {}, el("span", {}, el("span", { class: "swatch", style: `background:${b.colorCss}` }), `Pencil ${i}`),
        el("button", { class: "btn small", onclick: () => app.select(null) }, "✕")),
      el("div", { class: "kv" },
        el("span", {}, "state"), el("span", {}, b.moving ? `moving, ${fmt(b.speed_mms, 1)} mm/s` : (b.held_still ? "at rest" : "settling")),
        el("span", {}, "tilt"), el("span", {}, `${fmt(b.elevation_deg, 1)}° from flat`),
        el("span", {}, "height"), el("span", {}, b.floor.gap_mm <= 0.3
          ? `on the desk, lowest point its ${b.floor.at.region}`
          : `${fmt(b.floor.gap_mm, 1)} mm above the desk, lowest point its ${b.floor.at.region}`),
        el("span", {}, "resting on"), el("span", {}, rest),
        el("span", {}, "carrying"), el("span", {}, b.carrying.length ? b.carrying.map((c) => `pencil ${c}`).join(", ") : "—"),
        el("span", {}, "spin"), el("span", {}, `${fmt(b.spin_rads, 2)} rad/s`)),
      el("div", { class: "btns" },
        btn("Frame", () => app.camera.frame(i), "small"),
        btn(app.camera.following === i ? "Stop following" : "Follow", () => app.camera.follow(app.camera.following === i ? null : i), "small"),
        btn("Lift 5 cm", () => app.probe("lift", i, { height: 0.05 }), "small"),
        btn("Nudge", () => app.nudge(i), "small")));
  };
  app.onSelect(paintCard);
  setInterval(() => { if (!card.hidden && !root.classList.contains("hidden")) paintCard(); }, 500);

  // --------------------------------------------------------- labels
  const labels = $("labels");
  const labelEls = [];
  const paintLabels = () => {
    if (!app.display.get("labels") || root.classList.contains("hidden")) { labels.replaceChildren(); labelEls.length = 0; return; }
    const n = app.bodyCount;
    while (labelEls.length < n) { const d = el("div", { class: "label" }); labels.append(d); labelEls.push(d); }
    while (labelEls.length > n) labelEls.pop().remove();
    // place, then push any label that would sit on another one downwards
    const placed = [];
    const order = Array.from({ length: n }, (_, i) => i).map((i) => ({ i, p: app.project(i) }))
      .sort((a, b) => a.p.y - b.p.y);
    for (const { i, p } of order) {
      const d = labelEls[i];
      d.hidden = !p.visible;
      if (!p.visible) continue;
      const b = app.bodyInfo(i);
      d.textContent = `Pencil ${i} · ${fmt(b.elevation_deg, 0)}° · ${b.moving ? fmt(b.speed_mms, 0) + " mm/s" : "rest"}`;
      d.classList.toggle("sel", app.selected === i);
      const w = d.offsetWidth || 96, h = (d.offsetHeight || 20) + 4;
      let y = p.y;
      for (let guard = 0; guard < 8; guard++) {
        const hit = placed.find((q) => Math.abs(q.x - p.x) < (q.w + w) / 2 && Math.abs(q.y - y) < h);
        if (!hit) break;
        y = hit.y + h;
      }
      placed.push({ x: p.x, y, w });
      d.style.left = `${p.x}px`; d.style.top = `${y}px`;
    }
  };
  app.onFrame(paintLabels);

  // ------------------------------------------------------------ keys
  addEventListener("keydown", (e) => {
    if (e.target && /INPUT|SELECT|TEXTAREA/.test(e.target.tagName)) return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === " ") { e.preventDefault(); app.paused ? app.play() : app.pause(); }
    else if (e.key === ".") app.step();
    else if (e.key === "h" || e.key === "H") root.classList.toggle("hidden");
    else if (e.key === "?") $("help").hidden = !$("help").hidden;
    else if (e.key === "n" || e.key === "N") { const i = app.spawn(); toast(`pencil ${i} dropped`); }
    else if (e.key === "Escape") { $("help").hidden = true; app.select(null); }
  });

  return { toast, showTab };
}
