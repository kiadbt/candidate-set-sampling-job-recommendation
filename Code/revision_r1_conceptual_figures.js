#!/usr/bin/env node
"use strict";

// Creates the two non-data figures for the BDCC major revision. Text labels are
// primary; small outline icons are navigational cues. Arrowheads use fixed
// user-space dimensions and stop outside boxes.

const fs = require("fs");
const path = require("path");
const sharp = require("sharp");

const C = {
  navy: "#17365D", blue: "#4E79A7", teal: "#2A9D8F",
  orange: "#E38D26", red: "#C44E52", dark: "#202124",
  mid: "#5F6368", line: "#A7ADB4", lightBlue: "#EAF2F8",
  lightTeal: "#E8F5F1", lightOrange: "#FFF3E3",
  lightRed: "#FBEAEC", lightGray: "#F4F6F8",
};

function esc(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function svgStart(width, height) {
  return [`<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">`,
    `<rect width="${width}" height="${height}" fill="#fff"/>`,
    `<style>text{font-family:Arial,Helvetica,sans-serif;fill:${C.dark}} .panel{font-size:48px;font-weight:700;fill:${C.navy}} .title{font-size:38px;font-weight:700} .body{font-size:34px} .note{font-size:31px;fill:${C.mid}}</style>`,
    `<defs><marker id="arrow" markerWidth="12" markerHeight="12" refX="10.8" refY="6" orient="auto" markerUnits="userSpaceOnUse"><path d="M0 0 L12 6 L0 12 Z" fill="context-stroke"/></marker></defs>`];
}

function multiline(parts, x, y, lines, cls, gap, anchor = "start") {
  const tspans = lines.map((line, i) => `<tspan x="${x}" dy="${i === 0 ? 0 : gap}">${esc(line)}</tspan>`).join("");
  parts.push(`<text class="${cls}" x="${x}" y="${y}" text-anchor="${anchor}">${tspans}</text>`);
}

function icon(parts, kind, x, y, s, color) {
  const stroke = `fill="none" stroke="${color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"`;
  if (kind === "calendar") {
    parts.push(`<rect x="${x}" y="${y}" width="${s}" height="${0.78 * s}" ${stroke}/>`);
    parts.push(`<line x1="${x}" y1="${y + 0.30 * s}" x2="${x + s}" y2="${y + 0.30 * s}" ${stroke}/>`);
    parts.push(`<line x1="${x + 0.25 * s}" y1="${y - 0.08 * s}" x2="${x + 0.25 * s}" y2="${y + 0.16 * s}" ${stroke}/>`);
    parts.push(`<line x1="${x + 0.72 * s}" y1="${y - 0.08 * s}" x2="${x + 0.72 * s}" y2="${y + 0.16 * s}" ${stroke}/>`);
  } else if (kind === "catalog") {
    parts.push(`<ellipse cx="${x + s / 2}" cy="${y + 0.16 * s}" rx="${s / 2}" ry="${0.14 * s}" ${stroke}/>`);
    parts.push(`<path d="M${x} ${y + 0.16 * s} V${y + 0.70 * s} C${x} ${y + 0.87 * s},${x + s} ${y + 0.87 * s},${x + s} ${y + 0.70 * s} V${y + 0.16 * s}" ${stroke}/>`);
    parts.push(`<path d="M${x} ${y + 0.42 * s} C${x} ${y + 0.59 * s},${x + s} ${y + 0.59 * s},${x + s} ${y + 0.42 * s}" ${stroke}/>`);
  } else if (kind === "filter") {
    parts.push(`<path d="M${x} ${y} H${x + s} L${x + 0.62 * s} ${y + 0.42 * s} V${y + 0.82 * s} L${x + 0.39 * s} ${y + 0.67 * s} V${y + 0.42 * s} Z" ${stroke}/>`);
  } else if (kind === "rank") {
    [0.38, 0.65, 0.88].forEach((f, i) => parts.push(`<rect x="${x + i * 0.34 * s}" y="${y + (0.90 - f) * s}" width="${0.20 * s}" height="${f * s}" ${stroke}/>`));
  } else if (kind === "check") {
    parts.push(`<path d="M${x} ${y + 0.50 * s} L${x + 0.36 * s} ${y + 0.84 * s} L${x + s} ${y + 0.08 * s}" ${stroke}/>`);
  } else if (kind === "measure") {
    parts.push(`<path d="M${x} ${y + 0.86 * s} V${y + 0.12 * s} M${x} ${y + 0.86 * s} H${x + s} M${x + 0.12 * s} ${y + 0.68 * s} L${x + 0.44 * s} ${y + 0.34 * s} L${x + 0.71 * s} ${y + 0.57 * s} L${x + 0.96 * s} ${y + 0.18 * s}" ${stroke}/>`);
  }
}

function box(parts, x, y, w, h, title, lines, fill, edge, iconKind) {
  parts.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="18" fill="${fill}" stroke="${edge}" stroke-width="3"/>`);
  if (iconKind) icon(parts, iconKind, x + 22, y + 24, 38, edge);
  multiline(parts, x + (iconKind ? 76 : 26), y + 49, [title], "title", 0);
  multiline(parts, x + 26, y + 95, lines, "body", 42);
}

function arrow(parts, x1, y1, x2, y2, color = C.mid, dashed = false, curve = null) {
  const dash = dashed ? ` stroke-dasharray="10 8"` : "";
  if (curve) {
    parts.push(`<path d="M${x1} ${y1} Q${curve[0]} ${curve[1]} ${x2} ${y2}" fill="none" stroke="${color}" stroke-width="4"${dash} marker-end="url(#arrow)"/>`);
  } else {
    parts.push(`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="4"${dash} marker-end="url(#arrow)"/>`);
  }
}

async function writeFigure(outDir, stem, parts, width, height) {
  const svg = `${parts.join("")}\n</svg>`;
  const svgPath = path.join(outDir, `${stem}.svg`);
  const pngPath = path.join(outDir, `${stem}.png`);
  fs.mkdirSync(outDir, { recursive: true });
  fs.writeFileSync(svgPath, svg, "utf8");
  await sharp(Buffer.from(svg), { density: 600 })
    .resize({ width: 4800 })
    .png({ compressionLevel: 9 })
    .withMetadata({ density: 600 })
    .toFile(pngPath);
  return { svg: svgPath, png: pngPath };
}

async function interventionFigure(outDir) {
  const W = 2400, H = 1280;
  const p = svgStart(W, H);
  p.push(`<text class="panel" x="45" y="75">A  Temporal eligibility comparison</text>`);
  box(p, 60, 125, 430, 350, "Query at t(q)", ["Fixed query and", "observed relevant jobs", "with relevance horizon"], C.lightBlue, C.blue, "calendar");
  box(p, 570, 125, 470, 350, "Temporal rule", ["CareerBuilder: active", "at cutoff", "JTH: created by", "query date"], C.lightTeal, C.teal, "filter");
  box(p, 1120, 95, 440, 200, "Reference catalog", ["Protocol-eligible jobs"], C.lightTeal, C.teal, "catalog");
  box(p, 1120, 325, 440, 235, "Future-inclusive catalog", ["Reference catalog plus", "future-opening jobs"], C.lightRed, C.red, "catalog");
  box(p, 1660, 160, 680, 330, "Fixed comparison", ["Same queries and targets", "Same fields and model parameters", "Same shared-job scores", "Same metrics and tie rule"], C.lightOrange, C.orange, "measure");
  arrow(p, 490, 300, 570, 300);
  arrow(p, 1040, 260, 1120, 195);
  arrow(p, 1040, 340, 1120, 440, C.red);
  arrow(p, 1560, 195, 1660, 270);
  arrow(p, 1560, 440, 1660, 375, C.red);

  p.push(`<line x1="40" y1="625" x2="2360" y2="625" stroke="${C.line}" stroke-width="2"/>`);
  p.push(`<text class="panel" x="45" y="700">B  Candidate-set intervention within the temporal reference</text>`);
  box(p, 60, 760, 430, 330, "Eligible catalog E(q)", ["Observed relevant set", "R(q) plus eligible", "alternatives"], C.lightTeal, C.teal, "catalog");
  box(p, 590, 730, 470, 235, "Exact evaluation", ["Rank every job in E(q)"], C.lightBlue, C.blue, "rank");
  box(p, 590, 1000, 470, 235, "Sampled evaluation", ["R(q) plus sampled", "eligible alternatives", "All targets retained"], C.lightOrange, C.orange, "filter");
  box(p, 1180, 825, 500, 310, "Fixed scoring basis", ["Same model scores", "Same metrics and cutoffs", "Deterministic job-ID ties"], C.lightGray, C.mid, "check");
  box(p, 1810, 825, 530, 310, "Agreement outcomes", ["Metric bias", "Paired effects", "Model order and top-model set"], C.lightTeal, C.teal, "measure");
  arrow(p, 490, 880, 590, 845);
  arrow(p, 490, 970, 590, 1115, C.orange);
  arrow(p, 1060, 845, 1180, 925);
  arrow(p, 1060, 1115, 1180, 1035, C.orange);
  arrow(p, 1680, 980, 1810, 980);
  return writeFigure(outDir, "figure_01_temporal_and_candidate_interventions", p, W, H);
}

async function calibrationFigure(outDir) {
  const W = 2400, H = 1150;
  const p = svgStart(W, H);
  p.push(`<text class="panel" x="45" y="75">Decision-oriented calibration of sampled evaluation</text>`);
  p.push(`<text class="note" x="45" y="125">The acceptable candidate set depends on the claim that must be preserved.</text>`);
  const xs = [45, 520, 995, 1470, 1945];
  box(p, xs[0], 180, 400, 330, "1  Define", ["Temporal reference", "Models and metrics", "Decision to preserve"], C.lightBlue, C.blue, "calendar");
  box(p, xs[1], 180, 400, 330, "2  Calibrate", ["Representative", "exact-catalog", "calibration cohort"], C.lightTeal, C.teal, "catalog");
  box(p, xs[2], 180, 400, 330, "3  Sample", ["Nested sizes", "Declared sampler", "Multiple seeds", "All targets retained"], C.lightOrange, C.orange, "filter");
  box(p, xs[3], 180, 400, 330, "4  Measure", ["Bias and paired effect", "Top-model set", "Order, rank, stability"], C.lightGray, C.mid, "measure");
  box(p, xs[4], 180, 410, 330, "5  Test", ["Predeclare tolerances", "for the intended claim"], C.lightTeal, C.teal, "check");
  for (let i = 0; i < 4; i++) arrow(p, xs[i] + 400, 345, xs[i + 1], 345);

  p.push(`<text class="note" x="1760" y="585" text-anchor="middle" font-weight="700" fill="${C.teal}">All criteria pass</text>`);
  arrow(p, 2150, 510, 1770, 695, C.teal, false, [2050, 640]);
  box(p, 1370, 710, 800, 280, "6A  Use the smallest passing set", ["Validate on the exact-catalog cohort", "Report every calibration criterion"], C.lightTeal, C.teal, "check");

  p.push(`<text class="note" x="860" y="585" text-anchor="middle" font-weight="700" fill="${C.red}">Any criterion fails</text>`);
  arrow(p, 2070, 510, 820, 695, C.red, false, [1440, 640]);
  box(p, 200, 710, 920, 280, "6B  Increase evidence or narrow the claim", ["Use a larger set or exact evaluation", "State which decision was not preserved"], C.lightRed, C.red, "filter");
  p.push(`<text class="note" x="1200" y="1090" text-anchor="middle" font-style="italic">No universal candidate count, fraction, or correction is assumed.</text>`);
  return writeFigure(outDir, "figure_05_decision_oriented_calibration", p, W, H);
}

async function main() {
  const outIndex = process.argv.indexOf("--output-dir");
  if (outIndex < 0 || !process.argv[outIndex + 1]) {
    throw new Error("Usage: revision_r1_conceptual_figures.js --output-dir PATH");
  }
  const outDir = path.resolve(process.argv[outIndex + 1]);
  const outputs = [await interventionFigure(outDir), await calibrationFigure(outDir)];
  process.stdout.write(`${JSON.stringify(outputs, null, 2)}\n`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
