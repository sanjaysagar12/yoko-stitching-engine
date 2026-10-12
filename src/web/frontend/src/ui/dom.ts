// Tiny DOM helper: h("div", { class: "x" }, "text", child).
type Attrs = Record<string, string | number | boolean | undefined | ((ev: Event) => void)>;

export function h<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  attrs: Attrs = {},
  ...children: Array<Node | string | null | undefined>
): HTMLElementTagNameMap[K] {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === false) continue;
    if (typeof v === "function") el.addEventListener(k.replace(/^on/, ""), v);
    else if (k === "class") el.className = String(v);
    else if (v === true) el.setAttribute(k, "");
    else el.setAttribute(k, String(v));
  }
  for (const c of children) if (c !== null && c !== undefined) el.append(c);
  return el;
}

export function replaceChildren(el: HTMLElement, ...nodes: Node[]): void {
  el.replaceChildren(...nodes);
}
