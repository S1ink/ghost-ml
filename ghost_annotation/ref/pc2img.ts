import { Message } from "./types";

export const inputs = ["/multiscan/lidar_scan"];
export const output = "/foxglove/lidar_image";

type ImageMsg = Message<"sensor_msgs/Image">;
type PointCloudMsg = Message<"sensor_msgs/PointCloud2">;

type Config = {
  output_width: number;
  azimuth_offset: number;
  io_fn: string;
  output_minmax: number[];
  encoding: string;
  keep_closest: boolean;
  max_gap_interpolation: number;
};

const encodingInfo: Record<
  string,
  { code: string; bppx: number; norm: boolean; as_bytes: boolean }
> = {
  rgb: { code: "rgb8", bppx: 3, norm: true, as_bytes: true },
  float_norm: { code: "32FC1", bppx: 4, norm: true, as_bytes: false },
  float_raw: { code: "32FC1", bppx: 4, norm: false, as_bytes: false },
};

// ─── PointCloud2 datatype constants ──────────────────────────────────────────
const DT = {
  INT8: 1,
  UINT8: 2,
  INT16: 3,
  UINT16: 4,
  INT32: 5,
  UINT32: 6,
  FLOAT32: 7,
  FLOAT64: 8,
} as const;

// ─── Helpers ─────────────────────────────────────────────────────────────────

function getIntegerFromEnd(str: string): number {
  const match = str.match(/\d+$/);
  if (match) {
    // Specify radix 10 for decimal parsing
    return parseInt(match[0], 10);
  }
  return 0;
}

/** Read one value from a DataView at byte offset using the PointCloud2 datatype ID. */
function readScalar(
  view: DataView,
  byteOffset: number,
  datatype: number,
  le: boolean,
): number {
  switch (datatype) {
    case DT.FLOAT32:
      return view.getFloat32(byteOffset, le);
    case DT.FLOAT64:
      return view.getFloat64(byteOffset, le);
    case DT.INT8:
      return view.getInt8(byteOffset);
    case DT.UINT8:
      return view.getUint8(byteOffset);
    case DT.INT16:
      return view.getInt16(byteOffset, le);
    case DT.UINT16:
      return view.getUint16(byteOffset, le);
    case DT.INT32:
      return view.getInt32(byteOffset, le);
    case DT.UINT32:
      return view.getUint32(byteOffset, le);
    default:
      return NaN;
  }
}

/**
 * Jet colormap.  t ∈ [0, 1]  →  [R, G, B] ∈ [0, 255]
 * Blue = low value, Red = high value.
 */
function jet(t: number): [number, number, number] {
  const clamp = (v: number) => Math.max(0, Math.min(255, Math.round(v * 255)));
  return [
    clamp(Math.min(4 * t - 1.5, -4 * t + 4.5)),
    clamp(Math.min(4 * t - 0.5, -4 * t + 3.5)),
    clamp(Math.min(4 * t + 0.5, -4 * t + 2.5)),
  ];
}

/**
 * Fills blank rows in `valuePx`, per column, by linearly interpolating
 * between the nearest filled row above and below. Blank runs longer than
 * `maxGap` rows are left untouched. Edge runs (no filled row on one side)
 * are also left untouched, since there is nothing to interpolate towards.
 */
function interpolateColumnGaps(
  valuePx: Float32Array,
  outW: number,
  outH: number,
  maxGap: number,
): void {
  for (let col = 0; col < outW; col++) {
    let prevRow = -1;
    let prevVal = NaN;

    for (let row = 0; row < outH; row++) {
      const idx = row * outW + col;
      const val = valuePx[idx];
      if (isNaN(val)) continue;

      if (prevRow >= 0) {
        const gap = row - prevRow - 1;
        if (gap > 0 && gap <= maxGap) {
          const span = row - prevRow;
          for (let r = prevRow + 1; r < row; r++) {
            const t = (r - prevRow) / span;
            valuePx[r * outW + col] = prevVal + (val - prevVal) * t;
          }
        }
      }
      prevRow = row;
      prevVal = val;
    }
  }
}

// ─── Main entry point ─────────────────────────────────────────────────────────
export default function script(event: any, globals: any): ImageMsg | undefined {
  // 1. assemble primary inputs
  const cloud = event.message as PointCloudMsg;
  const procIdx = Math.max(0, getIntegerFromEnd(output) - 1);
  const configs = globals.pc2img_config.imgs;

  // 2. extract config if valid
  if (!Array.isArray(configs) || configs.length <= procIdx) return;
  const config = configs[procIdx] as Config;

  // 3. extract pointcloud fields and validate xyz exist
  const fieldsMap = new Map<string, { offset: number; datatype: number }>();
  for (const f of cloud.fields) {
    fieldsMap.set(f.name, { offset: f.offset, datatype: f.datatype });
  }

  const xf = fieldsMap.get("x");
  const yf = fieldsMap.get("y");
  const zf = fieldsMap.get("z");
  if (xf == null || yf == null || zf == null) return;

  // 4. extract polar data, filter which points are valid
  const data = cloud.data;
  const pointStep = cloud.point_step;
  const le = !cloud.is_bigendian;
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  const numInputPoints = Math.floor(data.byteLength / pointStep);
  if (numInputPoints < 1) return;

  const azimuths = new Float32Array(numInputPoints);
  const elevations = new Float32Array(numInputPoints);
  const ranges = new Float32Array(numInputPoints);
  const bases = new Uint32Array(numInputPoints);
  let numPoints = 0;

  let minAz = Infinity,
    maxAz = -Infinity;
  let minEl = Infinity,
    maxEl = -Infinity;
  let minRange = Infinity,
    maxRange = -Infinity;

  for (let i = 0; i < numInputPoints; i++) {
    const base = i * pointStep;
    const x = readScalar(view, base + xf.offset, xf.datatype, le);
    const y = readScalar(view, base + yf.offset, yf.datatype, le);
    const z = readScalar(view, base + zf.offset, zf.datatype, le);

    if (!isFinite(x) || !isFinite(y) || !isFinite(z)) continue;

    const xyDistSqrd = x * x + y * y;
    const range = Math.sqrt(xyDistSqrd + z * z);

    if (range < 1e-4) continue;

    const azimuth =
      ((((Math.atan2(y, x) + Math.PI + config.azimuth_offset) % (2 * Math.PI)) +
        2 * Math.PI) %
        (2 * Math.PI)) -
      Math.PI;
    const elevation = Math.atan2(z, Math.sqrt(xyDistSqrd));

    if (azimuth < minAz) minAz = azimuth;
    if (azimuth > maxAz) maxAz = azimuth;
    if (elevation < minEl) minEl = elevation;
    if (elevation > maxEl) maxEl = elevation;
    if (range < minRange) minRange = range;
    if (range > maxRange) maxRange = range;

    azimuths[numPoints] = azimuth;
    elevations[numPoints] = elevation;
    ranges[numPoints] = range;
    bases[numPoints] = base;
    numPoints++;
  }

  if (numPoints < 1) return;

  // 5. compute outputs
  let outVals: Float32Array, minVal: number, maxVal: number;
  if (config.io_fn.length < 1) {
    // 5a. simple reassignment if io fn is empty
    outVals = ranges;
    minVal = minRange;
    maxVal = maxRange;
  } else {
    // 5b. build io function AST and gather input fields
    const ioExpr = new Expression(config.io_fn);
    const ioVars = ioExpr.getVariableNames();

    // 5c. extract dependant fields if they exist
    const exprVarBuffers = new Map<string, Float32Array>();
    const rangeEquivFields = new Set(["depth", "range"]);
    const rangeSpan = Math.max(maxRange - minRange, 1e-6);
    let autoRangeKey = null;
    for (let i = 0; i < ioVars.length; i++) {
      const dep = ioVars[i];
      const auto = dep.includes("@");
      const stripped = dep.replaceAll("@", "");

      if (rangeEquivFields.has(dep)) {
        exprVarBuffers.set(dep, ranges);
        continue;
      }
      if (auto && rangeEquivFields.has(stripped)) {
        if (autoRangeKey === null) {
          const autoRange = new Float32Array(numPoints);
          ranges.forEach(
            (v, i, a) => (autoRange[i] = (v - minRange) / rangeSpan),
          );
          exprVarBuffers.set(dep, autoRange);
          autoRangeKey = dep;
        } else {
          const buff = exprVarBuffers.get(autoRangeKey);
          if (buff) {
            exprVarBuffers.set(dep, buff);
          }
        }
        continue;
      }

      const field = fieldsMap.get(stripped);
      if (field === undefined) return;

      const buffer = new Float32Array(numPoints);
      let min = Infinity,
        max = -Infinity;
      for (let p = 0; p < numPoints; p++) {
        const v = readScalar(view, bases[p] + field.offset, field.datatype, le);
        if (v < min) min = v;
        if (v > max) max = v;
        buffer[p] = v;
      }
      if (auto) {
        const span = Math.max(max - min, 1e-6);
        buffer.forEach((v, i, a) => (buffer[i] = (v - min) / span));
      }
      exprVarBuffers.set(dep, buffer);
    }

    // 5d. compute outputs by evaluating the io function
    outVals = new Float32Array(numPoints);
    minVal = Infinity;
    maxVal = -Infinity;
    for (let i = 0; i < numPoints; i++) {
      const val = ioExpr.evaluate((dep) => {
        const buffer = exprVarBuffers.get(dep);
        return buffer ? buffer[i] : 0;
      });

      if (val < minVal) minVal = val;
      if (val > maxVal) maxVal = val;
      outVals[i] = val;
    }
  }

  // 6. compute image size
  const azSpan = Math.max(maxAz - minAz, 1e-6);
  const elSpan = Math.max(maxEl - minEl, 1e-6);
  const aspect = azSpan / elSpan;

  let outW: number, outH: number;
  if (config.output_width > 0) {
    outW = config.output_width;
    outH = Math.floor(outW / aspect);
  } else {
    outH = Math.max(
      16,
      Math.min(1024, Math.round(Math.sqrt(numPoints / aspect))),
    );
    outW = Math.max(16, Math.min(4096, Math.round(outH * aspect)));
  }

  // 7. construct image buffer with raw values
  const valSpan = Math.max(maxVal - minVal, 1e-6);
  const encoding = encodingInfo[config.encoding];
  const rawImage = new Float32Array(outW * outH).fill(NaN);
  const nearestRangeImage = config.keep_closest
    ? new Float32Array(outW * outH).fill(Infinity)
    : null;

  for (let i = 0; i < numPoints; i++) {
    const u = (azimuths[i] - minAz) / azSpan;
    const v = (elevations[i] - minEl) / elSpan;

    if (u < 0 || u >= 1 || v < 0 || v >= 1) continue;

    const col = outW - 1 - Math.floor(u * outW);
    const row = Math.floor((1.0 - v) * outH);

    if (col < 0 || col >= outW || row < 0 || row >= outH) continue;

    const pixIdx = row * outW + col;

    if (nearestRangeImage) {
      const r = ranges[i];
      if (r >= nearestRangeImage[pixIdx]) continue;
      nearestRangeImage[pixIdx] = r;
    }

    rawImage[pixIdx] = outVals[i];
  }

  // 8. fill missing rows if configured to do so
  if (config.max_gap_interpolation != 0) {
    interpolateColumnGaps(
      rawImage,
      outW,
      outH,
      config.max_gap_interpolation > 0
        ? config.max_gap_interpolation
        : Infinity,
    );
  }

  // 9. resolve nomalization
  let normBase: number, normRange: number;
  if (
    config.output_minmax.length >= 2 &&
    !(config.output_minmax[0] == config.output_minmax[1])
  ) {
    normBase = config.output_minmax[0];
    normRange = config.output_minmax[1] - normBase;
  } else {
    normBase = minVal;
    normRange = valSpan;
  }

  // 10. fill final image
  const rowStep = outW * encoding.bppx;
  const image = encoding.as_bytes
    ? new Uint8Array(outH * rowStep)
    : new Float32Array(outH * outW);
  for (let i = 0; i < outW * outH; i++) {
    const raw = rawImage[i];
    if (!encoding.norm) {
      image[i] = isNaN(raw) ? 0 : raw;
      continue;
    }

    if (isNaN(raw)) continue;

    const t = Math.max(0.0, Math.min(1.0, (raw - normBase) / normRange));
    if (encoding.code === "rgb8") {
      image.set(jet(t), i * 3);
    } else {
      image[i] = t;
    }
  }

  return {
    header: cloud.header,
    height: outH,
    width: outW,
    encoding: encoding.code,
    is_bigendian: 0,
    step: rowStep,
    data: image instanceof Uint8Array ? image : new Uint8Array(image.buffer),
  };
}

/**
 * arith-expr — lightweight arithmetic expression parser/evaluator.
 *
 * Supports + - * / ^ (with standard precedence; ^ is right-associative),
 * parentheses, floating point numbers (including scientific notation), and
 * {variableName} references resolved through a pluggable VariableProvider.
 *
 * Grammar (precedence low -> high):
 *   expression := term (("+" | "-") term)*
 *   term       := unary (("*" | "/") unary)*
 *   unary      := ("+" | "-") unary | power
 *   power      := primary ("^" unary)?
 *   primary    := NUMBER | VARIABLE | "(" expression ")"
 *
 * `^` is right-associative (2^3^2 == 2^(3^2) == 512), and unary minus binds
 * looser than `^` on its operand, matching conventional math notation
 * (-2^2 == -4, not 4).
 *
 * Single-file edition — drop this anywhere in your project and import from
 * it directly. Everything below is self-contained; no external dependencies.
 */

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/** Thrown for lexical or grammatical problems while parsing an expression string. */
export class ExpressionSyntaxError extends Error {
  public readonly position?: number;

  constructor(message: string, position?: number) {
    super(message);
    this.name = "ExpressionSyntaxError";
    this.position = position;
    Object.setPrototypeOf(this, ExpressionSyntaxError.prototype);
  }
}

/** Thrown for problems that occur while evaluating an already-parsed AST. */
export class EvaluationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "EvaluationError";
    Object.setPrototypeOf(this, EvaluationError.prototype);
  }
}

// ---------------------------------------------------------------------------
// AST
// ---------------------------------------------------------------------------

export type BinaryOperator = "+" | "-" | "*" | "/" | "^";
export type UnaryOperator = "+" | "-";

export interface NumberNode {
  kind: "Number";
  value: number;
}

export interface VariableNode {
  kind: "Variable";
  name: string;
}

export interface UnaryOpNode {
  kind: "UnaryOp";
  operator: UnaryOperator;
  operand: ASTNode;
}

export interface BinaryOpNode {
  kind: "BinaryOp";
  operator: BinaryOperator;
  left: ASTNode;
  right: ASTNode;
}

export type ASTNode = NumberNode | VariableNode | UnaryOpNode | BinaryOpNode;

// ---------------------------------------------------------------------------
// Lexer
// ---------------------------------------------------------------------------

export enum TokenType {
  Number = "Number",
  Plus = "Plus",
  Minus = "Minus",
  Star = "Star",
  Slash = "Slash",
  Caret = "Caret",
  LParen = "LParen",
  RParen = "RParen",
  Variable = "Variable",
  EOF = "EOF",
}

export interface Token {
  type: TokenType;
  /** Raw text for operators/parens; numeric literal text for Number; variable name for Variable. */
  value: string;
  /** Character offset in the source string where this token starts. */
  position: number;
}

// Matches integers, decimals, and an optional exponent: 3, 3.14, .5, 2.5e-3
const NUMBER_RE = /^(\d+\.\d+|\.\d+|\d+)([eE][+-]?\d+)?/;

const SINGLE_CHAR_TOKENS: Partial<Record<string, TokenType>> = {
  "+": TokenType.Plus,
  "-": TokenType.Minus,
  "*": TokenType.Star,
  "/": TokenType.Slash,
  "^": TokenType.Caret,
  "(": TokenType.LParen,
  ")": TokenType.RParen,
};

/**
 * Converts a raw expression string into a flat list of tokens, ending in EOF.
 * Throws ExpressionSyntaxError on unrecognized characters, malformed numbers,
 * or unterminated `{...}` variable references.
 */
export function tokenize(input: string): Token[] {
  const tokens: Token[] = [];
  const n = input.length;
  let i = 0;

  while (i < n) {
    const ch = input[i];

    if (ch === " " || ch === "\t" || ch === "\n" || ch === "\r") {
      i++;
      continue;
    }

    const simpleType = SINGLE_CHAR_TOKENS[ch];
    if (simpleType !== undefined) {
      tokens.push({ type: simpleType, value: ch, position: i });
      i++;
      continue;
    }

    if (ch === "{") {
      const start = i;
      let j = i + 1;
      while (j < n && input[j] !== "}") j++;
      if (j >= n) {
        throw new ExpressionSyntaxError(
          `Unterminated variable reference starting at position ${start}`,
          start,
        );
      }
      let name = input.slice(i + 1, j).trim();
      if (name.length === 0) {
        throw new ExpressionSyntaxError(
          `Empty variable name at position ${start}`,
          start,
        );
      }
      if (name.includes("@")) {
        name = "@" + name.replaceAll("@", "");
      }
      tokens.push({ type: TokenType.Variable, value: name, position: start });
      i = j + 1;
      continue;
    }

    if ((ch >= "0" && ch <= "9") || ch === ".") {
      const match = NUMBER_RE.exec(input.slice(i));
      if (!match) {
        throw new ExpressionSyntaxError(
          `Invalid number literal at position ${i}`,
          i,
        );
      }
      tokens.push({ type: TokenType.Number, value: match[0], position: i });
      i += match[0].length;
      continue;
    }

    throw new ExpressionSyntaxError(
      `Unexpected character '${ch}' at position ${i}`,
      i,
    );
  }

  tokens.push({ type: TokenType.EOF, value: "", position: n });
  return tokens;
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

/**
 * Parses an arithmetic expression string into an AST.
 *
 * Supports +, -, *, /, ^ (right-associative), parentheses, floating point
 * numbers, and {variableName} references. Throws ExpressionSyntaxError on
 * malformed input (unexpected tokens, unbalanced parens, trailing input).
 */
export function parse(input: string): ASTNode {
  const tokens = tokenize(input);
  let pos = 0;

  const peek = (): Token => tokens[pos];
  const advance = (): Token => tokens[pos++];
  const check = (type: TokenType): boolean => peek().type === type;

  function expect(type: TokenType, message: string): Token {
    if (!check(type)) {
      const t = peek();
      const found = t.type === TokenType.EOF ? "end of input" : `'${t.value}'`;
      throw new ExpressionSyntaxError(
        `${message}, but found ${found} at position ${t.position}`,
        t.position,
      );
    }
    return advance();
  }

  function parseExpression(): ASTNode {
    let node = parseTerm();
    while (check(TokenType.Plus) || check(TokenType.Minus)) {
      const opToken = advance();
      const operator: BinaryOperator =
        opToken.type === TokenType.Plus ? "+" : "-";
      node = { kind: "BinaryOp", operator, left: node, right: parseTerm() };
    }
    return node;
  }

  function parseTerm(): ASTNode {
    let node = parseUnary();
    while (check(TokenType.Star) || check(TokenType.Slash)) {
      const opToken = advance();
      const operator: BinaryOperator =
        opToken.type === TokenType.Star ? "*" : "/";
      node = { kind: "BinaryOp", operator, left: node, right: parseUnary() };
    }
    return node;
  }

  function parseUnary(): ASTNode {
    if (check(TokenType.Plus) || check(TokenType.Minus)) {
      const opToken = advance();
      return {
        kind: "UnaryOp",
        operator: opToken.type === TokenType.Plus ? "+" : "-",
        operand: parseUnary(),
      };
    }
    return parsePower();
  }

  function parsePower(): ASTNode {
    const base = parsePrimary();
    if (check(TokenType.Caret)) {
      advance();
      // Recursing through parseUnary (rather than parsePower) lets exponents
      // carry their own sign, e.g. 2^-2, while the chained parsePower call
      // inside that recursion still yields right-associativity for 2^3^2.
      const exponent = parseUnary();
      return { kind: "BinaryOp", operator: "^", left: base, right: exponent };
    }
    return base;
  }

  function parsePrimary(): ASTNode {
    const t = peek();

    if (t.type === TokenType.Number) {
      advance();
      return { kind: "Number", value: parseFloat(t.value) };
    }

    if (t.type === TokenType.Variable) {
      advance();
      return { kind: "Variable", name: t.value };
    }

    if (t.type === TokenType.LParen) {
      advance();
      const inner = parseExpression();
      expect(TokenType.RParen, "Expected closing parenthesis");
      return inner;
    }

    const found = t.type === TokenType.EOF ? "end of input" : `'${t.value}'`;
    throw new ExpressionSyntaxError(
      `Expected a number, variable, or '(' but found ${found} at position ${t.position}`,
      t.position,
    );
  }

  const ast = parseExpression();
  expect(TokenType.EOF, "Unexpected trailing input");
  return ast;
}

// ---------------------------------------------------------------------------
// Evaluator
// ---------------------------------------------------------------------------

/**
 * The core interface for supplying variable values to an evaluation.
 * Implement this directly for custom sources (databases, caches, live
 * sensors, etc). For convenience, evaluate()/Expression#evaluate() also
 * accept a plain object, a Map, or a lookup function, and will normalize
 * them into this interface automatically (see toVariableProvider).
 */
export interface VariableProvider {
  get(name: string): number;
}

/** Anything that can be normalized into a VariableProvider. */
export type VariableSource =
  | VariableProvider
  | Record<string, number>
  | Map<string, number>
  | ((name: string) => number);

function isVariableProvider(
  source: VariableSource,
): source is VariableProvider {
  return (
    typeof source === "object" &&
    source !== null &&
    !(source instanceof Map) &&
    typeof (source as VariableProvider).get === "function"
  );
}

/** Normalizes any supported VariableSource into a VariableProvider. */
export function toVariableProvider(source: VariableSource): VariableProvider {
  if (typeof source === "function") {
    return { get: source };
  }
  if (source instanceof Map) {
    return {
      get(name: string): number {
        if (!source.has(name)) {
          throw new EvaluationError(`Missing value for variable '${name}'`);
        }
        return source.get(name) as number;
      },
    };
  }
  if (isVariableProvider(source)) {
    return source;
  }
  const record = source as Record<string, number>;
  return {
    get(name: string): number {
      if (!(name in record)) {
        throw new EvaluationError(`Missing value for variable '${name}'`);
      }
      return record[name];
    },
  };
}

/**
 * Evaluates an AST to a number. `variables` may be omitted only if the
 * expression contains no {variable} references.
 */
export function evaluate(node: ASTNode, variables?: VariableSource): number {
  const provider =
    variables !== undefined ? toVariableProvider(variables) : undefined;

  function evalNode(n: ASTNode): number {
    switch (n.kind) {
      case "Number":
        return n.value;

      case "Variable": {
        if (!provider) {
          throw new EvaluationError(
            `Expression references variable '${n.name}' but no variable source was supplied`,
          );
        }
        return provider.get(n.name);
      }

      case "UnaryOp": {
        const value = evalNode(n.operand);
        return n.operator === "-" ? -value : value;
      }

      case "BinaryOp": {
        const left = evalNode(n.left);
        const right = evalNode(n.right);
        switch (n.operator) {
          case "+":
            return left + right;
          case "-":
            return left - right;
          case "*":
            return left * right;
          case "/":
            if (right === 0) {
              throw new EvaluationError("Division by zero");
            }
            return left / right;
          case "^":
            return Math.pow(left, right);
        }
        // Unreachable: the switch above covers every BinaryOperator.
        throw new EvaluationError(
          `Unknown operator '${String((n as { operator: unknown }).operator)}'`,
        );
      }

      default:
        // Unreachable: the outer switch covers every ASTNode kind.
        throw new EvaluationError(`Unknown AST node: ${JSON.stringify(n)}`);
    }
  }

  return evalNode(node);
}

// ---------------------------------------------------------------------------
// Expression (parse once, evaluate many times)
// ---------------------------------------------------------------------------

/**
 * A parsed arithmetic expression, ready to be evaluated repeatedly
 * (e.g. against many different variable bindings) without re-parsing.
 *
 * Example:
 *   const expr = new Expression("2 * ({x} + 1) ^ 2 - {y}");
 *   expr.getVariableNames();        // ["x", "y"]
 *   expr.evaluate({ x: 3, y: 5 });  // 27
 */
export class Expression {
  public readonly source: string;
  public readonly ast: ASTNode;

  constructor(source: string) {
    this.source = source;
    this.ast = parse(source);
  }

  /** Evaluates this expression. Omit `variables` only if it has no {var} references. */
  evaluate(variables?: VariableSource): number {
    return evaluate(this.ast, variables);
  }

  /** Distinct variable names referenced by this expression, in order of first appearance. */
  getVariableNames(): string[] {
    const seen = new Set<string>();
    const names: string[] = [];

    const walk = (node: ASTNode): void => {
      switch (node.kind) {
        case "Variable":
          if (!seen.has(node.name)) {
            seen.add(node.name);
            names.push(node.name);
          }
          break;
        case "UnaryOp":
          walk(node.operand);
          break;
        case "BinaryOp":
          walk(node.left);
          walk(node.right);
          break;
        case "Number":
          break;
      }
    };

    walk(this.ast);
    return names;
  }

  toString(): string {
    return this.source;
  }
}

/** Convenience factory, equivalent to `new Expression(source)`. */
export function parseExpression(source: string): Expression {
  return new Expression(source);
}

/** One-shot parse + evaluate, for when you don't need to reuse the AST. */
export function evaluateExpression(
  source: string,
  variables?: VariableSource,
): number {
  return new Expression(source).evaluate(variables);
}
