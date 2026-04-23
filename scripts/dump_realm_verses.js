/**
 * Dump STTM Desktop's Realm `Verse` table to JSON for building our
 * order_id -> Realm Verse.ID and shabad-synthetic-id -> Realm ShabadID maps.
 *
 * Run:
 *   npm i realm
 *   node scripts/dump_realm_verses.js > data/realm_verses.json
 *
 * Env overrides (optional):
 *   STTM_REALM_PATH    — path to sttmdesktop-evergreen-v2.realm
 *   STTM_REALM_SCHEMA  — path to realm-schema-evergreen.json
 *
 * Defaults target macOS STTM Desktop's userData dir.
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
  console.error('Set STTM_REALM_PATH or install STTM Desktop first.');
  process.exit(1);
}
if (!fs.existsSync(schemaPath)) {
  console.error(`Schema file not found: ${schemaPath}`);
  process.exit(1);
}

// Modern `realm` npm refuses to open older file format versions read-only
// (it would need to upgrade the file in place, which requires write access).
// We don't want to mutate STTM's production DB, so copy to a tmp path and
// let the opener upgrade that copy instead.
const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'sttm-realm-'));
const workRealm = path.join(tmpDir, 'sttmdesktop-evergreen-v2.realm');
fs.copyFileSync(realmPath, workRealm);
// Lock / note auxiliary files sit next to the .realm if they exist.
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

// STTM Desktop ships schemas written in Realm JS v6/v10 shorthand
// ("date?", "int?", "Verse[]", etc.). The modern `realm` npm (v12+) rejects
// that and wants the fully-expanded `{type, objectType?, optional?}` form.
// Normalize each property so the dump script works against any realm version.
function expandShorthand(typeStr) {
  // Returns an object {type, objectType?, optional?} for a shorthand string, or null
  // if it's already a simple primitive name with no decoration.
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

(async () => {
  const realm = await Realm.open({
    path: workRealm,
    schema: schema.schemas,
    schemaVersion: schema.schemaVersion,
  });

  // Stream as a JSON array of compact objects. We only need:
  //   ID         — Realm Verse.ID (what STTM expects as verseId)
  //   Gurmukhi   — AnvaadLipi ASCII, matches our sqlite lines.gurmukhi verbatim
  //   ShabadIDs  — Realm ShabadID(s) this verse belongs to
  //   SourceID   — "G"/"D"/"B"/"N"/"A"/"S"/"R" (for disambiguation)
  const verses = realm.objects('Verse');
  process.stdout.write('[');
  let first = true;
  for (const v of verses) {
    if (!first) process.stdout.write(',');
    first = false;
    const shabadIds = [];
    if (v.Shabads) {
      for (const s of v.Shabads) shabadIds.push(s.ShabadID);
    }
    process.stdout.write(JSON.stringify({
      i: v.ID,
      g: v.Gurmukhi,
      pg: v.PageNo,
      s: shabadIds,
      src: v.Source ? v.Source.SourceID : null,
    }));
  }
  process.stdout.write(']');
  realm.close();
})().catch((err) => {
  console.error(err);
  process.exit(2);
});
