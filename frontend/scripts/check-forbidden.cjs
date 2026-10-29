/**
 * Verify prohibited interface language and visual tokens in source and the production build.
 * Strictly enforces Global Rules and Project Brief constraints.
 */

const fs = require('fs');
const path = require('path');
const ts = require('typescript');
const postcss = require('postcss');

const SRC_DIR = path.resolve(__dirname, '../src');
const DIST_DIR = path.resolve(__dirname, '../dist');

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
  if (!fs.existsSync(dir)) {
    throw new Error(`Required scan directory does not exist: ${dir}`);
  }
  const files = fs.readdirSync(dir);
  for (const file of files) {
    const fullPath = path.join(dir, file);
    const stat = fs.statSync(fullPath);
    if (stat.isDirectory()) {
      walkDir(fullPath, fileList);
    } else if (/\.(tsx?|jsx?|css|html|mjs|js)$/.test(file)) {
      fileList.push(fullPath);
    }
  }
  return fileList;
}

let violationsCount = 0;
let allFiles;
try {
  allFiles = [...walkDir(SRC_DIR), ...walkDir(DIST_DIR)];
} catch (error) {
  console.error(`[FAIL] ${error.message}`);
  process.exit(1);
}

console.log(`[CHECK] Scanning ${allFiles.length} source and production-build files for user-facing copy and colors...`);

function report(filePath, message) {
  const relativePath = path.relative(path.resolve(__dirname, '..'), filePath);
  console.error(`[VIOLATION] ${relativePath}: ${message}`);
  violationsCount++;
}

function scanCopy(text, filePath, context) {
  for (const pattern of FORBIDDEN_WORDS) {
    if (pattern.test(text)) {
      report(filePath, `forbidden user-facing copy (${pattern}): ${context}`);
    }
  }
  for (const pattern of FORBIDDEN_COLORS) {
    if (pattern.test(text)) {
      report(filePath, `forbidden design term (${pattern}): ${context}`);
    }
  }
  if (EMOJI_PATTERN.test(text)) {
    report(filePath, `forbidden emoji: ${context}`);
  }
}

function scanHexColors(text, filePath, context) {
  const hexMatches = text.match(/#[a-f0-9]{3,6}\b/gi) || [];
  for (const hex of hexMatches) {
    if (isForbiddenColorHex(hex)) {
      report(filePath, `forbidden purple-range color ${hex}: ${context}`);
    }
  }
}

function visitJavaScriptStrings(sourceFile, filePath) {
  function visit(node) {
    if (ts.isStringLiteral(node) || ts.isNoSubstitutionTemplateLiteral(node)) {
      scanCopy(node.text, filePath, `string literal at line ${sourceFile.getLineAndCharacterOfPosition(node.getStart(sourceFile)).line + 1}`);
      scanHexColors(node.text, filePath, `string literal at line ${sourceFile.getLineAndCharacterOfPosition(node.getStart(sourceFile)).line + 1}`);
    } else if (ts.isTemplateExpression(node)) {
      scanCopy(node.head.text, filePath, `template literal at line ${sourceFile.getLineAndCharacterOfPosition(node.getStart(sourceFile)).line + 1}`);
      scanHexColors(node.head.text, filePath, `template literal at line ${sourceFile.getLineAndCharacterOfPosition(node.getStart(sourceFile)).line + 1}`);
      for (const span of node.templateSpans) {
        scanCopy(span.literal.text, filePath, `template literal at line ${sourceFile.getLineAndCharacterOfPosition(node.getStart(sourceFile)).line + 1}`);
        scanHexColors(span.literal.text, filePath, `template literal at line ${sourceFile.getLineAndCharacterOfPosition(node.getStart(sourceFile)).line + 1}`);
      }
    } else if (ts.isJsxText(node)) {
      scanCopy(node.text, filePath, `JSX text at line ${sourceFile.getLineAndCharacterOfPosition(node.getStart(sourceFile)).line + 1}`);
    }
    ts.forEachChild(node, visit);
  }
  visit(sourceFile);
}

function scanCss(filePath, content) {
  const root = postcss.parse(content, { from: filePath });
  root.walkDecls((declaration) => {
    scanCopy(declaration.value, filePath, `CSS declaration "${declaration.prop}"`);
    scanHexColors(declaration.value, filePath, `CSS declaration "${declaration.prop}"`);
  });
}

function scanHtml(filePath, content) {
  scanCopy(content.replace(/<[^>]*>/g, ' '), filePath, 'HTML text');
  scanHexColors(content, filePath, 'HTML color');
}

for (const filePath of allFiles) {
  const relativePath = path.relative(path.resolve(__dirname, '..'), filePath);
  const content = fs.readFileSync(filePath, 'utf-8');
  const extension = path.extname(filePath).toLowerCase();
  const isBuiltVendor = filePath.startsWith(DIST_DIR) && /^vendor[-.]/i.test(path.basename(filePath));

  if (extension === '.css') {
    scanCss(filePath, content);
  } else if (extension === '.html') {
    scanHtml(filePath, content);
  } else if (['.ts', '.tsx', '.js', '.jsx', '.mjs'].includes(extension) && !isBuiltVendor) {
    const scriptKind = extension === '.tsx' || extension === '.jsx' ? ts.ScriptKind.TSX : ts.ScriptKind.JS;
    const sourceFile = ts.createSourceFile(filePath, content, ts.ScriptTarget.Latest, true, scriptKind);
    visitJavaScriptStrings(sourceFile, filePath);
  }
}

if (violationsCount > 0) {
  console.error(`\nFAILED: Found ${violationsCount} forbidden word/color/emoji violation(s). Build aborted.`);
  process.exit(1);
} else {
  console.log('PASS: No forbidden user-facing copy, colors, or emojis found in source and application build files.\n');
  process.exit(0);
}
