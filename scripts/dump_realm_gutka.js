/**
 * Dump STTM Desktop's Realm `Banis_Shabad` table for exact Sundar Gutka
 * controller pointers.
 *
 * STTM bani-controller payloads use `verseId` as `Banis_Shabad.ID`
 * (stored in the UI as `crossPlatformId`), not the normal `Verse.ID`.
 *
 * Run:
 *   npm i realm
 *   node scripts/dump_realm_gutka.js > data/realm_gutka_lines.json
 *
 * Env overrides:
 *   STTM_REALM_PATH    — path to sttmdesktop-evergreen-v2.realm
 *   STTM_REALM_SCHEMA  — path to realm-schema-evergreen.json
 */
const path = require('path');
const os = require('os');
const fs = require('fs');
const Realm = require('realm');

const defaultRealm = path.join(
  os.homedir(),
  'Library', 'Application Support', 'SikhiToTheMax',
  'sttmdesktop-evergreen-v2.realm',
);
const defaultSchema = path.join(
  os.homedir(),
  'Library', 'Application Support', 'SikhiToTheMax',
  'realm-schema-evergreen.json',
);

const realmPath = process.env.STTM_REALM_PATH || defaultRealm;
const schemaPath = process.env.STTM_REALM_SCHEMA || defaultSchema;

if (!fs.existsSync(realmPath)) {
  console.error(`Realm file not found: ${realmPath}`);
  process.exit(1);
}
if (!fs.existsSync(schemaPath)) {
  console.error(`Schema file not found: ${schemaPath}`);
  process.exit(1);
}

const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'sttm-realm-'));
const workRealm = path.join(tmpDir, 'sttmdesktop-evergreen-v2.realm');
fs.copyFileSync(realmPath, workRealm);
for (const ext of ['.lock', '.management', '.note']) {
  const src = realmPath + ext;
  if (fs.existsSync(src)) {
    const dst = workRealm + ext;
    try { fs.cpSync(src, dst, { recursive: true }); } catch {}
  }
}
process.on('exit', () => {
  try { fs.rmSync(tmpDir, { recursive: true, force: true }); } catch {}
});

const rawSchema = JSON.parse(fs.readFileSync(schemaPath, 'utf8'));

function expandShorthand(typeStr) {
  const listMatch = typeStr.match(/^(.+)\[\]$/);
  if (listMatch) return { type: 'list', objectType: listMatch[1] };
  const dictMatch = typeStr.match(/^(.+)\{\}$/);
  if (dictMatch) return { type: 'dictionary', objectType: dictMatch[1] };
  const setMatch = typeStr.match(/^(.+)<>$/);
  if (setMatch) return { type: 'set', objectType: setMatch[1] };
  if (typeStr.endsWith('?')) {
    const base = typeStr.slice(0, -1);
    const primitives = new Set([
      'int', 'string', 'bool', 'float', 'double', 'date', 'data',
      'decimal128', 'objectId', 'uuid', 'mixed',
    ]);
    if (primitives.has(base)) return { type: base, optional: true };
    return { type: 'object', objectType: base, optional: true };
  }
  return null;
}

function expandProp(v) {
  if (typeof v === 'string') {
    return expandShorthand(v) || v;
  }
  if (v && typeof v === 'object') {
    const { type, ...rest } = v;
    if (typeof type === 'string') {
      const expanded = expandShorthand(type);
      if (expanded && typeof expanded === 'object') {
        return { ...expanded, ...rest };
      }
      return { type, ...rest };
    }
  }
  return v;
}

const schema = {
  schemaVersion: rawSchema.schemaVersion,
  schemas: rawSchema.schemas.map((s) => {
    const props = {};
    for (const [k, v] of Object.entries(s.properties)) {
      props[k] = expandProp(v);
    }
    return { ...s, properties: props };
  }),
};

function rowText(row) {
  if (row.Verse) return row.Verse.Gurmukhi || '';
  if (row.Custom) return row.Custom.Gurmukhi || '';
  return '';
}

function rowVerseId(row) {
  if (row.Verse && row.Verse.ID !== undefined) return row.Verse.ID;
  if (row.Custom && row.Custom.ID !== undefined) return row.Custom.ID;
  return null;
}

(async () => {
  const realm = await Realm.open({
    path: workRealm,
    schema: schema.schemas,
    schemaVersion: schema.schemaVersion,
  });

  const rows = realm.objects('Banis_Shabad').sorted('Seq');
  process.stdout.write('[');
  let first = true;
  for (const row of rows) {
    if (!first) process.stdout.write(',');
    first = false;
    process.stdout.write(JSON.stringify({
      cross_platform_id: row.ID,
      bani_id: row.Bani ? row.Bani.ID : null,
      seq: row.Seq,
      verse_id: rowVerseId(row),
      gurmukhi: rowText(row),
      existsSGPC: !!row.existsSGPC,
      existsMedium: !!row.existsMedium,
      existsTaksal: !!row.existsTaksal,
      existsBuddhaDal: !!row.existsBuddhaDal,
    }));
  }
  process.stdout.write(']');
  realm.close();
})().catch((err) => {
  console.error(err);
  process.exit(2);
});
