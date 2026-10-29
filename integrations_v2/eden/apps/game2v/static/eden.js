// SPDX-FileCopyrightText: Copyright (c) 2026 Vast AI
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Game2V browser additions, loaded after FlashDreams' own /app.js (a classic
// script, so its top-level `peer`, `controls`, `video` and `sendInput` are in
// scope here). Ported for FlashDreams v0.2.0 from the EdenLive fork's app.js and
// index.html hunks: the four-panel layout, the per-game HUD overlay, the prompt
// and event panels and the /ext extension loader. The game render arrives as the
// left half of a side-by-side stream, so it is pixel-paired with the generated
// frame (right half) instead of travelling as a second track.

(() => {
  const css = `
html, body { overflow: hidden; }
body { padding: clamp(.5rem, 1.5vw, 1.5rem); box-sizing: border-box; }
.layout { display: grid; grid-template-columns: minmax(0, 2.2fr) minmax(20rem, 1fr); grid-template-rows: repeat(3, minmax(0, 1fr)); gap: .75rem; width: 100%; height: calc(100vh - 2 * clamp(.5rem, 1.5vw, 1.5rem)); min-height: 30rem; }
.panel { background: #1b1b1b; border: 1px solid #333; border-radius: .5rem; overflow: hidden; display: flex; flex-direction: column; min-height: 0; }
.model { grid-row: 1 / span 3; }
.panel-title { flex: 0 0 auto; padding: .5rem .75rem; color: #aaa; font-size: .8rem; text-transform: uppercase; }
.stage { position: relative; flex: 1 1 auto; min-height: 0; }
/* ID selectors: they must win over core's own #video rule. */
#video, #game-video { position: absolute; display: block; object-fit: cover; }
#video { object-position: 100% 50%; }
#game-video { object-position: 0% 50%; }
#game-overlay { position: absolute; pointer-events: none; overflow: hidden; }
#game-overlay button, #game-overlay [data-game-action] { pointer-events: auto; }
#prompt, #events { flex: 1 1 auto; min-height: 0; padding: 1rem; white-space: pre-wrap; overflow-wrap: anywhere; overflow-y: auto; overflow-x: hidden; scrollbar-gutter: stable; scrollbar-width: thin; scrollbar-color: #555 transparent; }
#events:empty::before { content: "No events in this chunk."; color: #777; }
@media (max-width: 700px) { .layout { grid-template-columns: 1fr; grid-template-rows: minmax(16rem, 55vh) repeat(3, 11rem); height: auto; min-height: 0; } .model { grid-row: auto; } html, body { overflow: auto; } }
`;
  const style = document.createElement("style");
  style.textContent = css;
  document.head.append(style);

  const panel = (title, body, extraClass = "") => {
    const section = document.createElement("section");
    section.className = `panel ${extraClass}`.trim();
    const heading = document.createElement("div");
    heading.className = "panel-title";
    heading.textContent = title;
    section.append(heading, body);
    return section;
  };
  const stage = document.createElement("div");
  stage.className = "stage";
  const overlayHost = document.createElement("div");
  overlayHost.id = "game-overlay";
  overlayHost.hidden = true;
  stage.append(video, overlayHost);
  const promptBox = document.createElement("div");
  promptBox.id = "prompt";
  promptBox.textContent = "Waiting for the first chunk…";
  const eventsBox = document.createElement("div");
  eventsBox.id = "events";
  const gameStage = document.createElement("div");
  gameStage.className = "stage";
  const gameVideo = document.createElement("video");
  gameVideo.id = "game-video";
  gameVideo.autoplay = gameVideo.muted = gameVideo.playsInline = true;
  gameStage.append(gameVideo);
  const layout = document.createElement("main");
  layout.className = "layout";
  layout.append(
    panel("Generated model video", stage, "model"),
    panel("Scene prompt", promptBox),
    panel("Chunk events", eventsBox),
    panel("Game render", gameStage),
  );
  document.body.prepend(layout);

  // Both panels show the one stream; each crops its half of the canvas.
  const coreOnTrack = peer.ontrack;
  peer.ontrack = event => {
    coreOnTrack(event);
    gameVideo.srcObject = video.srcObject;
    gameVideo.play().catch(() => {});
  };

  // Fit an element to the largest centred box of the half-frame aspect ratio.
  const fit = (element, container) => {
    const box = container.getBoundingClientRect();
    if (!video.videoWidth || !video.videoHeight || !box.width || !box.height) {
      return null;
    }
    const aspect = video.videoWidth / 2 / video.videoHeight;
    const width = Math.min(box.width, box.height * aspect);
    const height = width / aspect;
    const rect = {left: (box.width - width) / 2, top: (box.height - height) / 2, width, height};
    element.style.left = `${rect.left}px`;
    element.style.top = `${rect.top}px`;
    element.style.width = `${rect.width}px`;
    element.style.height = `${rect.height}px`;
    return rect;
  };

  let overlayView = null;
  let overlaySlug = null;
  let overlayFrameId = -1;
  const layoutPanels = () => {
    fit(gameVideo, gameStage);
    const rect = fit(video, stage);
    if (rect === null) {
      overlayHost.hidden = true;
      return;
    }
    fit(overlayHost, stage);
    overlayHost.hidden = overlayView === null;
  };
  new ResizeObserver(layoutPanels).observe(layout);
  video.addEventListener("resize", layoutPanels);
  video.addEventListener("loadedmetadata", layoutPanels);

  const loadOverlay = async slug => {
    if (!slug || slug === overlaySlug) {
      return overlayView;
    }
    const shadow = overlayHost.shadowRoot ?? overlayHost.attachShadow({mode: "open"});
    shadow.replaceChildren();
    overlayView = null;
    overlaySlug = slug;
    overlayFrameId = -1;
    const sheet = document.createElement("link");
    sheet.rel = "stylesheet";
    sheet.href = `/games/${encodeURIComponent(slug)}/hud/overlay.css`;
    const mountRoot = document.createElement("div");
    mountRoot.style.cssText = "position:absolute;inset:0;";
    shadow.append(sheet, mountRoot);
    const module = await import(`/games/${encodeURIComponent(slug)}/hud/overlay.js`);
    if (overlaySlug !== slug) {
      return null;
    }
    overlayView = module.mount(mountRoot);
    mountRoot.addEventListener("click", event => {
      const action = event.target.closest("[data-game-action]")?.dataset.gameAction;
      if (!action) {
        return;
      }
      event.preventDefault();
      event.stopPropagation();
      sendInput({type: "client_action", channel: "game_ui", action});
    });
    layoutPanels();
    return overlayView;
  };

  const applyGameUi = payload => {
    window.dispatchEvent(new CustomEvent("game2v-ui", {detail: payload}));
    const slug = payload?.slug;
    const frameId = Number(payload?.frame_id);
    if (typeof slug !== "string" || !slug) {
      return;
    }
    if (Number.isFinite(frameId) && overlaySlug === slug && frameId === overlayFrameId) {
      return;
    }
    if (Number.isFinite(frameId)) {
      overlayFrameId = frameId;
    }
    loadOverlay(slug).then(view => {
      view?.update(payload.state);
      layoutPanels();
    }).catch(error => {
      console.warn("Unable to load the game overlay.", error);
    });
  };

  controls.addEventListener("message", event => {
    if (typeof event.data !== "string") {
      return;
    }
    let payload;
    try {
      payload = JSON.parse(event.data);
    } catch {
      return;
    }
    if (payload?.type === "prompt" && Array.isArray(payload.events)) {
      // One chunk's complete set; the next chunk's message replaces it.
      if (typeof payload.static === "string" && payload.static) {
        promptBox.textContent = payload.static;
      }
      eventsBox.replaceChildren(...payload.events.filter(line => typeof line === "string").map(line => {
        const item = document.createElement("div");
        item.textContent = line;
        return item;
      }));
    } else if (payload?.type === "game_ui") {
      applyGameUi(payload);
    }
  });

  (async () => {
    try {
      // The session registers its extensions before the server reports it
      // open; a page opened during model loading would otherwise see none.
      while (!(await fetch("/healthz", {cache: "no-store"})
        .then(response => (response.ok ? response.json() : null))
        .catch(() => null))?.open) {
        await new Promise(resolve => setTimeout(resolve, 250));
      }
      const manifest = await (await fetch("/extensions.json", {cache: "no-store"})).json();
      for (const src of manifest.scripts ?? []) {
        try {
          await import(src);
        } catch (error) {
          console.warn(`Unable to load extension script ${src}.`, error);
        }
      }
    } catch (error) {
      console.warn("Unable to load /extensions.json.", error);
    }
  })();
})();
