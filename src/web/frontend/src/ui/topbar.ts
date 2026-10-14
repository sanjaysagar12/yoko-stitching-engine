// Top bar: job drop-down, build progress, display toggles, save / load / export.
import type { Store } from "../state";
import type { Actions } from "./actions";
import { h } from "./dom";

export interface ExportDialogs {
  openAssembly(): void;
  openImages(): void;
}

export function mountTopbar(host: HTMLElement, store: Store, actions: Actions, dialogs: ExportDialogs): void {
  const select = h("select", { "data-testid": "job-select", "aria-label": "Job" });
  select.addEventListener("change", () => {
    if (select.value) actions.selectJob(select.value);
  });
  const status = h("span", { class: "status", "data-testid": "status" }, "");
  const bar = h("div", { class: "progress" }, h("div", { class: "progress-fill" }));
  const fill = bar.firstElementChild as HTMLElement;

  const toggle = (id: string, label: string, on: (v: boolean) => void): HTMLLabelElement => {
    const cb = h("input", { type: "checkbox", "data-testid": id });
    cb.addEventListener("change", () => on(cb.checked));
    return h("label", { class: "toggle" }, cb, label);
  };
  const opacity = h("input", { type: "range", min: 0.1, max: 1, step: 0.05, value: 0.5, "aria-label": "Avatar opacity" });
  opacity.addEventListener("input", () => store.set({ avatarOpacity: Number(opacity.value) }));
  const btn = (text: string, id: string, fn: () => void): HTMLButtonElement =>
    h("button", { class: "btn", "data-testid": id, onclick: fn }, text);

  const fileIn = h("input", { type: "file", accept: ".json,application/json", hidden: true, "data-testid": "import-assembly-input", "aria-label": "Assembly file" });
  fileIn.addEventListener("change", () => {
    const f = fileIn.files?.[0];
    fileIn.value = ""; // the same file can be chosen again
    if (f) void actions.importAssembly(f);
  });
  const importBtn = btn("Import assembly", "import-assembly", () => fileIn.click());
  importBtn.title = "Load an assembly.json: builds its job, makes the derived pieces, places and assembles (or drop the file on the viewer)";
  const exportAsm = btn("Export assembly", "export-assembly", dialogs.openAssembly);
  const exportImg = btn("Export images", "export-images", dialogs.openImages);
  const cancel = btn("Cancel", "cancel-busy", actions.cancelBusy);
  cancel.hidden = true;

  host.append(
    h("strong", { class: "title" }, "Pattern fit viewer"),
    h("label", { class: "job" }, "Job ", select),
    h("div", { class: "progress-wrap" }, bar, status),
    toggle("toggle-labels", "Labels", (v) => store.set({ showLabels: v })),
    toggle("toggle-volumes", "Volumes", (v) => store.set({ showVolumes: v })),
    h("label", { class: "toggle" }, "Avatar ", opacity),
    h("span", { class: "spacer" }),
    btn("Save", "save", actions.save),
    btn("Load", "load", actions.load),
    btn("Export OBJ", "export-obj", () => actions.exportScene("obj")),
    btn("Export GLB", "export-glb", () => actions.exportScene("glb")),
    importBtn,
    fileIn,
    exportAsm,
    exportImg,
    cancel,
  );

  const syncButtons = (): void => {
    const s = store.get();
    exportAsm.disabled = exportImg.disabled = !s.buildId;
    exportAsm.title = s.buildId ? "Download the layout as <garment>.assembly.json" : "Build a job first";
    exportImg.title = s.buildId ? "Render every view, the sections and the contact sheet into a zip" : "Build a job first";
    cancel.hidden = !s.cancel;
  };
  syncButtons();
  store.subscribe((s, changed) => {
    if (["buildId", "cancel"].some((k) => changed.includes(k as never))) syncButtons();
    if (changed.includes("jobs")) {
      select.replaceChildren(
        h("option", { value: "" }, "Choose a job..."),
        ...s.jobs.map((j) => h("option", { value: j.name }, j.name)),
      );
      select.value = s.job ?? "";
    }
    if (changed.includes("job")) select.value = s.job ?? "";
    if (changed.includes("busy") || changed.includes("job") || changed.includes("build")) {
      status.textContent = s.busy ? s.busy.text : s.build ? `ready: ${s.build.pieces.length} pieces, ${s.build.avatar.points.length} points` : "";
      bar.classList.toggle("active", Boolean(s.busy));
      fill.style.width = `${Math.round((s.busy?.progress ?? (s.build ? 1 : 0)) * 100)}%`;
    }
  });
}
