// What the UI can ask the application to do (implemented in main.ts).
import type { ArrangeOptions, Axis, PairMode, Placement } from "../types";

export interface CloneRequest {
  source: string;
  axis: Axis;
  id: string;
  /** Place the pair as soon as an anchor is known (the placement of the source, or the next point clicked). */
  pairNow: boolean;
  mode: PairMode;
}

export interface CopyRequest {
  source: string;
  /** Names of the new copies (one per copy). */
  ids: string[];
  /** ``now``: the first copy is placed on the next point clicked; ``opposite``: on the counterpart (``_L`` / ``_R``) of
   * the source's point, same wrap / offset / reference / rotation; ``none``: not placed. */
  place: "none" | "now" | "opposite";
}

export type ViewSide = "front" | "back" | "left" | "right" | "top";

export interface Actions {
  selectJob(name: string): void;
  selectPiece(name: string | null): void;
  choosePoint(name: string): void;
  updatePlacement(piece: string, patch: Partial<Placement>): void;
  removePlacement(piece: string): void;
  /** A named point of the selected piece was clicked in the inspector: use it as the reference. */
  setReference(piece: string, name: string): void;
  openClone(piece: string): void;
  cloneAndPair(req: CloneRequest): Promise<void>;
  openCopy(piece: string): void;
  copyPieces(req: CopyRequest): Promise<void>;
  deleteDerived(id: string): void;
  placePair(source: string, derivedId: string, mode: PairMode): void;
  /** Emphasise the two points of a mirrored-anchor pair (null to clear). */
  hoverPair(source: string, mode: PairMode | null): void;
  save(): void;
  load(): void;
  exportScene(format: "obj" | "glb"): void;
  viewAvatar(): void;
  viewPieces(): void;
  viewSide(side: ViewSide): void;
  /** Auto-arrange the placed pieces on offset body shells (POST /api/arrange). */
  arrange(): void;
  /** Back to the placements as made (wrap modes), without the arrangement. */
  resetArrange(): void;
  setArrangeOptions(patch: Partial<ArrangeOptions>): void;
  setMeshLines(on: boolean): void;
  /** Show or hide one piece (the eye button of the piece list). */
  toggleHidden(piece: string): void;
  /** Frame the place of a violation and highlight its pieces. */
  focusViolation(index: number): void;
}
