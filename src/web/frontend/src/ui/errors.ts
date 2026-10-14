// Error panel: structured errors from the backend (code, message, piece / tool), never a stack trace.
import type { Store } from "../state";
import type { Actions } from "./actions";
import { h } from "./dom";

export function mountErrors(host: HTMLElement, store: Store, actions?: Actions): void {
  const panel = h("div", { class: "error-panel", "data-testid": "error-panel", hidden: true });
  host.append(panel);
  store.subscribe((s, changed) => {
    if (!changed.includes("error")) return;
    const e = s.error;
    panel.hidden = !e;
    if (!e) return panel.replaceChildren();
    panel.replaceChildren(h("strong", {}, e.code), h("span", {}, ` ${e.message}`));
    if (e.piece) panel.append(h("span", { class: "meta" }, `  piece: ${e.piece}`));
    if (e.tool) panel.append(h("span", { class: "meta" }, `  tool: ${e.tool}`));
    if (e.problems && e.problems.length > 0) {
      panel.append(h("ul", { class: "problems", "data-testid": "error-problems" }, ...e.problems.map((p) => h("li", {}, h("strong", {}, p.code), ` ${p.message}`))));
      if (actions && e.problems.some((p) => p.code === "unknown_piece")) {
        panel.append(h("button", { class: "btn small", "data-testid": "import-skip-missing", onclick: () => actions.retryImportSkipping() }, "Import without the missing pieces"));
      }
    }
    panel.append(h("button", { class: "btn small", onclick: () => store.set({ error: null }) }, "Dismiss"));
  });
}
