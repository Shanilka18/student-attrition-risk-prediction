/**
 * Verification script checking that no forbidden design tokens, words, or emojis exist in frontend/src.
 * Strictly enforces Global Rules and Project Brief constraints.
 */

const fs = require('fs');
const path = require('path');

const SRC_DIR = path.resolve(__dirname, '../src');

// Forbidden copy list (PROJECT_BRIEF §8, §10, User Rules)
const FORBIDDEN_WORDS = [
  /\bprototype\b/i,
  /\bdemo\b/i,
  /\bbeta\b/i,
  /\bmock\b/i,
  /\bplaceholder\b/i,
  /\bsmart\b/i,
  /\bmagic\b/i,
  /\bpowered\s+by\s+ai\b/i,
  /\bai-powered\b/i,
];

// Forbidden colors & design tropes (PROJECT_BRIEF §10, Phase 12 Rules)
const FORBIDDEN_COLORS = [
  /\bpurple\b/i,
  /\bindigo\b/i,
  /\bviolet\b/i,
  /\bmagenta\b/i,
  /\bneon\b/i,
  /\bgradient\b/i,
  /\bglassmorphism\b/i,
];

// Hex color parser to detect forbidden purple/indigo/violet/magenta range (H: 245° - 335°, S > 20%)
function isForbiddenColorHex(hex) {
  // Normalize 3-char hex to 6-char
  let cleanHex = hex.replace('#', '');
  if (cleanHex.length === 3) {
    cleanHex = cleanHex.split('').map(c => c + c).join('');
  }
  if (cleanHex.length !== 6) return false;

  const r = parseInt(cleanHex.substring(0, 2), 16) / 255;
  const g = parseInt(cleanHex.substring(2, 4), 16) / 255;
  const b = parseInt(cleanHex.substring(4, 6), 16) / 255;

  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const delta = max - min;

  if (delta === 0) return false; // Grayscale / achromatic

  let h = 0;
  if (max === r) {
    h = ((g - b) / delta) % 6;
  } else if (max === g) {
    h = (b - r) / delta + 2;
  } else {
    h = (r - g) / delta + 4;
  }
  h = Math.round(h * 60);
  if (h < 0) h += 360;

  const l = (max + min) / 2;
  const s = delta / (1 - Math.abs(2 * l - 1));

  // Purple / Indigo / Violet / Magenta range: 245° to 335°, with perceptible saturation > 20%
  return h >= 245 && h <= 335 && s > 0.20;
}

// Emoji pattern
const EMOJI_PATTERN = /[\u{1F300}-\u{1F9FF}\u{2600}-\u{26FF}\u{2700}-\u{27BF}]/u;

function walkDir(dir, fileList = []) {
  const files = fs.readdirSync(dir);
  for (const file of files) {
    const fullPath = path.join(dir, file);
    const stat = fs.statSync(fullPath);
    if (stat.isDirectory()) {
      walkDir(fullPath, fileList);
    } else if (/\.(tsx?|jsx?|css|html)$/.test(file)) {
      fileList.push(fullPath);
    }
  }
  return fileList;
}

let violationsCount = 0;
const allFiles = walkDir(SRC_DIR);

console.log(`[CHECK] Scanning ${allFiles.length} files in frontend/src for forbidden words and colors...`);

for (const filePath of allFiles) {
  const relativePath = path.relative(path.resolve(__dirname, '..'), filePath);
  const content = fs.readFileSync(filePath, 'utf-8');
  const lines = content.split('\n');

  lines.forEach((line, lineIdx) => {
    // 1. Check forbidden words
    for (const pattern of FORBIDDEN_WORDS) {
      if (pattern.test(line)) {
        console.error(`[VIOLATION] Forbidden word match (${pattern}): ${relativePath}:${lineIdx + 1}`);
        console.error(`   > ${line.trim()}`);
        violationsCount++;
      }
    }

    // 2. Check forbidden colors
    for (const pattern of FORBIDDEN_COLORS) {
      if (pattern.test(line)) {
        console.error(`[VIOLATION] Forbidden color/trope (${pattern}): ${relativePath}:${lineIdx + 1}`);
        console.error(`   > ${line.trim()}`);
        violationsCount++;
      }
    }

    // 3. Check forbidden hex colors (purple/indigo/violet/magenta)
    const hexMatches = line.match(/#[a-f0-9]{3,6}\b/gi) || [];
    for (const hex of hexMatches) {
      if (isForbiddenColorHex(hex)) {
        console.error(`[VIOLATION] Forbidden purple/indigo/violet/magenta hex (${hex}): ${relativePath}:${lineIdx + 1}`);
        console.error(`   > ${line.trim()}`);
        violationsCount++;
      }
    }

    // 4. Check emojis
    if (EMOJI_PATTERN.test(line)) {
      console.error(`[VIOLATION] Forbidden emoji: ${relativePath}:${lineIdx + 1}`);
      console.error(`   > ${line.trim()}`);
      violationsCount++;
    }
  });
}

if (violationsCount > 0) {
  console.error(`\nFAILED: Found ${violationsCount} forbidden word/color/emoji violation(s). Build aborted.`);
  process.exit(1);
} else {
  console.log('PASS: Zero forbidden words, colors, or emojis found in frontend/src.\n');
  process.exit(0);
}
