// Left panel: avatar measurement table (target / achieved / error, cm) and the axis convention.
import { formatCm, formatSigned } from "../lib";
import type { Store } from "../state";
import { h } from "./dom";

export const CONVENTION = "Metres. +x = wearer's left, +y up, +z front. Feet on y = 0.";

export function mountMeasurements(host: HTMLElement, store: Store): void {
  const body = h("div", { "data-testid": "measurements" });
  host.append(h("h2", {}, "Avatar"), h("p", { class: "convention", "data-testid": "convention" }, CONVENTION), body);
  store.subscribe((s, changed) => {
    if (!changed.includes("build")) return;
    const av = s.build?.avatar;
    if (!av) return body.replaceChildren();
    const rows = Object.entries(av.measurements.measurements).filter(([, r]) => r.target_cm !== null);
    body.replaceChildren(
      h(
        "p",
        { class: "meta" },
        `${av.bundle.model_type.toUpperCase()} ${av.bundle.model_file}, ${av.bundle.num_betas_used} shape components`,
      ),
      h(
        "table",
        { class: "mtable" },
        h("thead", {}, h("tr", {}, ...["", "target", "achieved", "error"].map((t) => h("th", {}, t)))),
        h(
          "tbody",
          {},
          ...rows.map(([name, r]) =>
            h(
              "tr",
              { title: r.definition },
              h("td", {}, name),
              h("td", {}, formatCm(r.target_cm)),
              h("td", {}, formatCm(r.post_correction_cm)),
              h("td", { class: Math.abs(r.post_error_cm ?? 0) > 0.2 ? "warn" : "" }, formatSigned(r.post_error_cm, 2)),
            ),
          ),
        ),
      ),
      h("p", { class: "meta" }, "cm. Hover a row for its definition."),
    );
  });
}
