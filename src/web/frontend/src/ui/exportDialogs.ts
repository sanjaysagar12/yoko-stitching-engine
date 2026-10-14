// Dialogs of the two exports: the assembly file (name, notes, solved poses, mode) and the image zip (what to include).
import type { Store } from "../state";
import type { Actions } from "./actions";
import { h } from "./dom";
import type { ExportDialogs } from "./topbar";

const NAME = /^[A-Za-z0-9_.-]+$/;

export function mountExportDialogs(store: Store, actions: Actions): ExportDialogs {
  const asmDialog = h("dialog", { class: "clone-dialog", "data-testid": "export-assembly-dialog", "aria-labelledby": "export-assembly-title" });
  const imgDialog = h("dialog", { class: "clone-dialog", "data-testid": "export-images-dialog", "aria-labelledby": "export-images-title" });
  document.body.append(asmDialog, imgDialog);

  const check = (label: string, id: string, on: boolean): [HTMLLabelElement, HTMLInputElement] => {
    const cb = h("input", { type: "checkbox", "data-testid": id });
    cb.checked = on;
    return [h("label", { class: "radio" }, cb, label), cb];
  };
  const nameField = (id: string, value: string): HTMLInputElement => h("input", { type: "text", value, "data-testid": id, "aria-label": "Garment name" });

  const openAssembly = (): void => {
    const s = store.get();
    if (!s.job) return;
    const garment = nameField("export-garment", s.assembly?.payload.garment ?? s.job);
    const notes = h("textarea", { rows: 4, "data-testid": "export-notes", "aria-label": "Notes" });
    notes.value = s.assembly?.payload.notes ?? "";
    const [solvedLabel, solved] = check("Include solved poses (reopens exactly as it is)", "export-solved", true);
    const mode = h(
      "select",
      { "data-testid": "export-mode", "aria-label": "Mode" },
      h("option", { value: "arrange" }, "arrange (all pieces together)"),
      h("option", { value: "incremental" }, "incremental (one piece at a time)"),
    );
    mode.value = s.assembly?.payload.mode ?? "arrange";
    const msg = h("div", { class: "warn", "data-testid": "export-assembly-msg" });
    const go = (): void => {
      const name = garment.value.trim();
      if (!NAME.test(name)) return void (msg.textContent = "The name may use letters, digits, _ . - only.");
      asmDialog.close();
      void actions.exportAssembly({ garment: name, notes: notes.value, includeSolved: solved.checked, mode: mode.value as "arrange" | "incremental" });
    };
    asmDialog.replaceChildren(
      h("h3", { id: "export-assembly-title" }, "Export assembly"),
      h("p", { class: "meta" }, `Job ${s.job}: ${s.placements.size} placements, ${s.derived.length} derived pieces. Saved as <garment>.assembly.json.`),
      h("label", { class: "row2" }, "Garment ", garment),
      h("label", { class: "row2" }, "Notes ", notes),
      solvedLabel,
      h("label", { class: "row2" }, "Mode ", mode),
      msg,
      h(
        "div",
        { class: "dialog-buttons" },
        h("button", { class: "btn", "data-testid": "export-assembly-cancel", onclick: () => asmDialog.close() }, "Cancel"),
        h("button", { class: "btn primary", "data-testid": "export-assembly-go", onclick: go }, "Download"),
      ),
    );
    asmDialog.showModal();
    garment.select();
  };

  const openImages = (): void => {
    const s = store.get();
    if (!s.buildId) return;
    const garment = nameField("images-garment", s.assembly?.payload.garment ?? s.job ?? "garment");
    const [c1, closeups] = check("Per-piece close-ups", "images-closeups", true);
    const [c2, heat] = check("Clearance heat maps", "images-heat", true);
    const [c3, verify] = check("Verification report (quick check; slow on a real body)", "images-verify", false);
    const [c4, assembly] = check("The assembly.json", "images-assembly", true);
    const [c5, viewport] = check("Also include viewport screenshots (what you see now)", "images-viewport", false);
    const msg = h("div", { class: "warn", "data-testid": "export-images-msg" });
    const go = (): void => {
      const name = garment.value.trim();
      if (!NAME.test(name)) return void (msg.textContent = "The name may use letters, digits, _ . - only.");
      imgDialog.close();
      void actions.exportImages({
        garment: name,
        viewport: viewport.checked,
        options: { closeups: closeups.checked, heat: heat.checked, verify: verify.checked, assembly: assembly.checked },
      });
    };
    imgDialog.replaceChildren(
      h("h3", { id: "export-images-title" }, "Export images"),
      h("p", { class: "meta" }, "Every angle, the contact sheet and the horizontal sections of the layout on screen, rendered on the server in real size (nothing moves), as <garment>_images.zip."),
      h("label", { class: "row2" }, "Garment ", garment),
      c1,
      c2,
      c3,
      c4,
      c5,
      msg,
      h(
        "div",
        { class: "dialog-buttons" },
        h("button", { class: "btn", "data-testid": "export-images-cancel", onclick: () => imgDialog.close() }, "Cancel"),
        h("button", { class: "btn primary", "data-testid": "export-images-go", onclick: go }, "Export"),
      ),
    );
    imgDialog.showModal();
    garment.select();
  };

  return { openAssembly, openImages };
}
