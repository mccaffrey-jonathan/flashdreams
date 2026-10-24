// SPDX-License-Identifier: Apache-2.0
/**
 * Game2V generation panel. Lists registered games, previews one in an iframe,
 * imports a finished GamePrototypeTurbo project, and asks the next model step
 * to switch the live proxy game.
 */
const API = "/ext/game2v";

const css = `
.game2v-panel{display:flex;flex-direction:column;gap:.6rem;padding:.75rem;min-height:0;height:100%;box-sizing:border-box;font:13px system-ui,sans-serif;color:#ddd}
.game2v-panel h2{margin:0;font:600 .75rem system-ui,sans-serif;letter-spacing:.08em;text-transform:uppercase;color:#aaa}
.game2v-row{display:flex;gap:.4rem;align-items:center;flex-wrap:wrap}
.game2v-panel select,.game2v-panel input,.game2v-panel button{background:#111;color:#eee;border:1px solid #444;border-radius:.3rem;padding:.35rem .55rem;font:inherit}
.game2v-panel button{cursor:pointer}
.game2v-panel button:disabled{opacity:.5;cursor:default}
.game2v-panel input{flex:1 1 8rem;min-width:0}
.game2v-status{color:#9ab;min-height:1.2em}
.game2v-status.error{color:#f88}
.game2v-preview{flex:1 1 auto;min-height:8rem;border:1px solid #333;border-radius:.4rem;background:#000}
.game2v-preview iframe{display:block;width:100%;height:100%;border:0;background:#000}
.game2v-meta{color:#888;white-space:pre-wrap}
`;

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "className") node.className = value;
    else if (key.startsWith("on") && typeof value === "function") node.addEventListener(key.slice(2).toLowerCase(), value);
    else if (value != null) node.setAttribute(key, value);
  }
  node.append(...children);
  return node;
}

async function json(path, options) {
  const response = await fetch(`${API}${path}`, {cache: "no-store", ...options});
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(body.error || response.statusText);
  }
  return body;
}

function mount() {
  const layout = document.querySelector("main.layout");
  if (!layout || document.querySelector(".game2v-host")) {
    return;
  }
  layout.style.gridTemplateRows = "repeat(4, minmax(0, 1fr))";
  const model = layout.querySelector(".model");
  if (model) model.style.gridRow = "1 / span 4";
  const host = el("section", {className: "panel game2v-host"});
  host.append(el("div", {className: "panel-title"}, "Prototype games"));
  const root = el("div", {className: "game2v-panel"});
  const style = document.createElement("style");
  style.textContent = css;
  document.head.append(style);
  host.append(root);
  layout.append(host);

  const select = el("select");
  const status = el("div", {className: "game2v-status"});
  const meta = el("div", {className: "game2v-meta"});
  const preview = el("iframe", {
    title: "Game preview",
    allow: "autoplay",
    sandbox: "allow-scripts allow-same-origin",
  });
  const project = el("input", {type: "text", placeholder: "Skill project path", spellcheck: "false"});
  const slug = el("input", {type: "text", placeholder: "slug", spellcheck: "false"});
  const play = el("iframe", {hidden: "hidden"});
  void play;

  const setStatus = (text, isError = false) => {
    status.textContent = text;
    status.classList.toggle("error", isError);
  };

  const fill = (payload) => {
    const current = payload.current;
    select.replaceChildren(...payload.games.map((game) => {
      const option = el("option", {value: game.slug}, `${game.title} (${game.slug})`);
      if (game.slug === current) option.selected = true;
      return option;
    }));
    const selected = payload.games.find((game) => game.slug === select.value);
    meta.textContent = selected
      ? `${selected.host === "standalone" ? "Imported" : "Hand-ported"}\n${(selected.controls || []).join("\n")}`
      : "";
    preview.src = `${API}/play/${encodeURIComponent(select.value)}`;
  };

  const refresh = async () => {
    try {
      fill(await json("/games"));
    } catch (error) {
      setStatus(error.message, true);
    }
  };

  const switchBtn = el("button", {
    type: "button",
    onClick: async () => {
      switchBtn.disabled = true;
      try {
        fill(await json("/switch", {
          method: "POST",
          headers: {"content-type": "application/json"},
          body: JSON.stringify({slug: select.value}),
        }));
        setStatus(`Next chunk uses ${select.value}.`);
      } catch (error) {
        setStatus(error.message, true);
      } finally {
        switchBtn.disabled = false;
      }
    },
  }, "Use as next proxy");

  const importBtn = el("button", {
    type: "button",
    onClick: async () => {
      importBtn.disabled = true;
      try {
        setStatus("Importing…");
        const payload = await json("/import", {
          method: "POST",
          headers: {"content-type": "application/json"},
          body: JSON.stringify({
            project: project.value.trim(),
            slug: slug.value.trim(),
            replace: true,
            allow_incomplete: true,
          }),
        });
        fill(payload);
        if (payload.imported) select.value = payload.imported;
        preview.src = `${API}/play/${encodeURIComponent(select.value)}`;
        setStatus(`Imported ${payload.imported}.`);
      } catch (error) {
        setStatus(error.message, true);
      } finally {
        importBtn.disabled = false;
      }
    },
  }, "Import project");

  select.addEventListener("change", () => {
    preview.src = `${API}/play/${encodeURIComponent(select.value)}`;
  });

  root.append(
    el("h2", {}, "Registered"),
    el("div", {className: "game2v-row"}, select, switchBtn),
    meta,
    el("div", {className: "game2v-preview"}, preview),
    el("h2", {}, "Import skill project"),
    el("div", {className: "game2v-row"}, project, slug, importBtn),
    status,
  );
  refresh();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", mount);
} else {
  mount();
}
