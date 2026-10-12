// Error panel: structured errors from the backend (code, message, piece / tool), never a stack trace.
import type { Store } from "../state";
import { h } from "./dom";

export function mountErrors(host: HTMLElement, store: Store): void {
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
    panel.append(h("button", { class: "btn small", onclick: () => store.set({ error: null }) }, "Dismiss"));
  });
}
