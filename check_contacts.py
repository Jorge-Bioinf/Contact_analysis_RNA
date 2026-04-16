#!/usr/bin/env python3
"""
check_contacts.py

Entity-aware RNA / ligand contact checker for mmCIF files.

Core ideas
----------
1. Polymer / non-polymer / branched / water comes from CIF entity metadata.
2. Polymer subtype comes from _entity_poly.type when available.
3. Polymer membership / unusual monomers are checked with:
   - _entity_poly_seq
   - _pdbx_poly_seq_scheme
4. True non-polymer instances come from:
   - _pdbx_entity_nonpoly
   - _pdbx_nonpoly_scheme
5. Branched entities are recognized separately.
6. _atom_site is used for coordinates and final structural matching.
7. _chem_comp is only annotation / validation, never the primary truth.

This prevents false ligand calls for polymer-associated modified monomers such as GTP.

Usage
-----
python check_contacts.py 9E5T
python check_contacts.py 9E5T -f 9e5t.cif
python check_contacts.py 4TS2 --show-all
python check_contacts.py 4TS2 --include-ions
python check_contacts.py 4TS2 --include-buffers
python check_contacts.py 4TS2 --include-branched
python check_contacts.py 4TS2 --threshold 4.5
python check_contacts.py 4TS2 -o contacts.tsv
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import sys
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from scipy.spatial import cKDTree


# -----------------------------------------------------------------------------
# Non-polymer / small-molecule subtyping helpers
# -----------------------------------------------------------------------------
ION_COMP_IDS = {
    "LI", "NA", "K", "RB", "CS",
    "MG", "CA", "SR", "BA",
    "MN", "FE", "CO", "NI", "CU", "ZN", "CD", "HG",
    "AL", "GA", "IN",
    "CL", "BR", "IOD",
    # additional ions / small inorganic anions
    "UNX", "BO4", "CO3", "PI", "BCT", "PER", "OXL",
    "MOO", "LCP", "LCO", "WO4", "PO4", "AZI", "SO4",
    "CYN", "NO2", "YT3", "2HP", "AUC", "MLI",
    "NO3", "BF4", "SUL", "PO3", "ALF", "NCO","FES"
}

BUFFER_SOLVENT_COMP_IDS = {
    "TRS", "MES", "HEP", "MOPS", "BIS", "BME", "DTT", "EDO", "EOH",
    "GOL", "MPD", "PEG", "PGE", "PG4", "PG5", "PG6", "IPA",
    "ACY", "FMT", "ACE", "CIT", "TLA", "ACT",
}

WATER_COMP_IDS = {"HOH", "DOD", "WAT"}


# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------
@dataclass
class EntityInfo:
    entity_id: str
    entity_type: str
    description: Optional[str] = None
    src_method: Optional[str] = None
    formula_weight: Optional[str] = None
    number_of_molecules: Optional[str] = None


@dataclass
class PolymerInfo:
    entity_id: str
    poly_type: Optional[str] = None
    strand_ids: List[str] = field(default_factory=list)
    seq_one_letter: Optional[str] = None
    seq_one_letter_can: Optional[str] = None
    nstd_linkage: Optional[str] = None
    nstd_monomer: Optional[str] = None


@dataclass
class ChemCompInfo:
    comp_id: str
    comp_type: Optional[str] = None
    name: Optional[str] = None
    mon_nstd_flag: Optional[str] = None
    pdbx_type: Optional[str] = None
    formula: Optional[str] = None
    formula_weight: Optional[str] = None


@dataclass
class AtomRecord:
    group_PDB: str
    atom_id: str
    element: str
    atom_name: str
    alt_id: str
    comp_id: str
    label_asym_id: str
    label_entity_id: str
    label_seq_id: Optional[str]
    auth_asym_id: str
    auth_seq_id: str
    auth_comp_id: str
    ins_code: str
    x: float
    y: float
    z: float


@dataclass
class StructuralInstance:
    entity_id: str
    entity_kind: str               # polymer / non-polymer / branched / water
    category: str                  # RNA / Ligand / Ion/Inorganic / Branched / Water ...
    comp_id: str
    label_asym_id: str
    auth_asym_id: str
    auth_seq_id: str
    description: Optional[str] = None
    chem_comp_type: Optional[str] = None
    source: str = ""
    warnings: List[str] = field(default_factory=list)


# -----------------------------------------------------------------------------
# CIF retrieval
# -----------------------------------------------------------------------------
def fetch_cif(pdb_id: str, output_dir: str = ".") -> Optional[str]:
    pdb_id = pdb_id.lower()
    path = os.path.join(output_dir, f"{pdb_id}.cif")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        print(f"  Using cached: {path}")
        return path

    url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    print(f"  Downloading: {url}")
    try:
        urllib.request.urlretrieve(url, path)
        return path
    except Exception as exc:
        print(f"  ERROR downloading {pdb_id}: {exc}")
        return None


# -----------------------------------------------------------------------------
# mmCIF parsing
# -----------------------------------------------------------------------------
def _clean_token(tok: Optional[str]) -> Optional[str]:
    if tok is None:
        return None
    tok = tok.strip()
    if tok in {"?", "."}:
        return None
    return tok


def tokenize_cif_text(text: str) -> List[str]:
    """
    Tokenize mmCIF-ish text while preserving semicolon-delimited multiline blocks.

    This is not a full standards-complete parser, but is robust enough for:
    - long multiline sequence/code fields
    - quoted tokens
    - regular loop rows
    """
    tokens: List[str] = []
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]

        if ch.isspace():
            i += 1
            continue

        # multiline semicolon text: only when ';' is at start of line
        if ch == ";" and (i == 0 or text[i - 1] == "\n"):
            i += 1
            start = i
            end = text.find("\n;", i)
            if end == -1:
                tokens.append(text[start:].rstrip("\n"))
                break
            tokens.append(text[start:end])
            i = end + 2
            continue

        if ch in {"'", '"'}:
            quote = ch
            i += 1
            start = i
            while i < n and text[i] != quote:
                i += 1
            tokens.append(text[start:i])
            if i < n:
                i += 1
            continue

        start = i
        while i < n and not text[i].isspace():
            i += 1
        tokens.append(text[start:i])

    return tokens


def parse_mmcif(cif_path: str) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """
    Lightweight mmCIF parser.
    Returns:
      loops: item_name -> list of column values
      singles: item_name -> scalar value
    """
    with open(cif_path, "r", encoding="utf-8") as handle:
        lines = handle.readlines()

    loops: Dict[str, List[str]] = {}
    singles: Dict[str, str] = {}

    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        s = line.strip()

        if not s or s == "#":
            i += 1
            continue

        if s == "loop_":
            i += 1
            colnames: List[str] = []
            while i < n and lines[i].strip().startswith("_"):
                colnames.append(lines[i].strip())
                i += 1

            row_lines: List[str] = []
            while i < n:
                cur = lines[i]
                stripped = cur.strip()
                if not stripped:
                    i += 1
                    continue
                if stripped == "#":
                    break
                if stripped == "loop_":
                    break
                if stripped.startswith("_"):
                    break
                row_lines.append(cur)
                i += 1

            if colnames:
                flat_tokens = tokenize_cif_text("".join(row_lines))
                ncols = len(colnames)

                rows: List[List[Optional[str]]] = []
                cursor = 0
                while cursor + ncols <= len(flat_tokens):
                    rows.append(flat_tokens[cursor: cursor + ncols])
                    cursor += ncols

                for cidx, cname in enumerate(colnames):
                    loops[cname] = [row[cidx] if cidx < len(row) else None for row in rows]
            continue

        if s.startswith("_"):
            # single item, maybe value on same line
            parts = shlex.split(s)
            if len(parts) >= 2:
                singles[parts[0]] = parts[1]
                i += 1
                continue

            # single item followed by multiline ; block
            if len(parts) == 1 and i + 1 < n:
                nxt = lines[i + 1]
                if nxt.startswith(";"):
                    block_lines: List[str] = []
                    i += 2
                    while i < n and not lines[i].startswith(";"):
                        block_lines.append(lines[i].rstrip("\n"))
                        i += 1
                    singles[parts[0]] = "\n".join(block_lines)
                    if i < n:
                        i += 1
                    continue

        i += 1

    return loops, singles


def get_loop_rows(loops: Dict[str, List[str]], prefix: str) -> List[Dict[str, Optional[str]]]:
    cols = [k for k in loops if k.startswith(prefix)]
    if not cols:
        return []

    lengths = {len(loops[c]) for c in cols}
    if len(lengths) != 1:
        raise ValueError(f"Inconsistent loop lengths for prefix {prefix}")

    nrows = lengths.pop()
    rows: List[Dict[str, Optional[str]]] = []
    for i in range(nrows):
        row: Dict[str, Optional[str]] = {}
        for col in cols:
            row[col[len(prefix):]] = _clean_token(loops[col][i])
        rows.append(row)
    return rows


def get_category_rows(
    loops: Dict[str, List[str]],
    singles: Dict[str, str],
    prefix: str,
) -> List[Dict[str, Optional[str]]]:
    """
    Supports either:
    - a looped category
    - a single-row category encoded as item/value pairs
    """
    rows = get_loop_rows(loops, prefix)
    if rows:
        return rows

    items = {
        key[len(prefix):]: _clean_token(value)
        for key, value in singles.items()
        if key.startswith(prefix)
    }
    if items:
        return [items]

    return []


def load_tables(cif_path: str) -> Dict[str, List[Dict[str, Optional[str]]]]:
    loops, singles = parse_mmcif(cif_path)
    return {
        "entity": get_category_rows(loops, singles, "_entity."),
        "entity_poly": get_category_rows(loops, singles, "_entity_poly."),
        "entity_poly_seq": get_category_rows(loops, singles, "_entity_poly_seq."),
        "pdbx_poly_seq_scheme": get_category_rows(loops, singles, "_pdbx_poly_seq_scheme."),
        "struct_asym": get_category_rows(loops, singles, "_struct_asym."),
        "entity_nonpoly": get_category_rows(loops, singles, "_pdbx_entity_nonpoly."),
        "nonpoly_scheme": get_category_rows(loops, singles, "_pdbx_nonpoly_scheme."),
        "entity_branch": get_category_rows(loops, singles, "_pdbx_entity_branch."),
        "branch_scheme": get_category_rows(loops, singles, "_pdbx_branch_scheme."),
        "chem_comp": get_category_rows(loops, singles, "_chem_comp."),
        "atom_site": get_category_rows(loops, singles, "_atom_site."),
    }


# -----------------------------------------------------------------------------
# Metadata builders
# -----------------------------------------------------------------------------
def build_entity_map(rows: List[Dict[str, Optional[str]]]) -> Dict[str, EntityInfo]:
    result: Dict[str, EntityInfo] = {}
    for row in rows:
        entity_id = row.get("id")
        entity_type = row.get("type")
        if not entity_id or not entity_type:
            continue
        result[entity_id] = EntityInfo(
            entity_id=entity_id,
            entity_type=entity_type,
            description=row.get("pdbx_description"),
            src_method=row.get("src_method"),
            formula_weight=row.get("formula_weight"),
            number_of_molecules=row.get("pdbx_number_of_molecules"),
        )
    return result


def build_polymer_map(rows: List[Dict[str, Optional[str]]]) -> Dict[str, PolymerInfo]:
    result: Dict[str, PolymerInfo] = {}
    for row in rows:
        entity_id = row.get("entity_id")
        if not entity_id:
            continue
        strand_ids = []
        strand_field = row.get("pdbx_strand_id")
        if strand_field:
            strand_ids = [x.strip() for x in strand_field.split(",") if x.strip()]
        result[entity_id] = PolymerInfo(
            entity_id=entity_id,
            poly_type=row.get("type"),
            strand_ids=strand_ids,
            seq_one_letter=row.get("pdbx_seq_one_letter_code"),
            seq_one_letter_can=row.get("pdbx_seq_one_letter_code_can"),
            nstd_linkage=row.get("nstd_linkage"),
            nstd_monomer=row.get("nstd_monomer"),
        )
    return result


def build_entity_poly_seq_map(
    rows: List[Dict[str, Optional[str]]]
) -> Dict[str, List[Dict[str, Optional[str]]]]:
    result: Dict[str, List[Dict[str, Optional[str]]]] = defaultdict(list)
    for row in rows:
        entity_id = row.get("entity_id")
        if entity_id:
            result[entity_id].append(row)
    return dict(result)


def build_poly_seq_scheme_map(
    rows: List[Dict[str, Optional[str]]]
) -> Dict[str, List[Dict[str, Optional[str]]]]:
    result: Dict[str, List[Dict[str, Optional[str]]]] = defaultdict(list)
    for row in rows:
        entity_id = row.get("entity_id")
        if entity_id:
            result[entity_id].append(row)
    return dict(result)


def build_struct_asym_map(rows: List[Dict[str, Optional[str]]]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for row in rows:
        asym_id = row.get("id")
        entity_id = row.get("entity_id")
        if asym_id and entity_id:
            result[asym_id] = entity_id
    return result


def build_nonpoly_entity_map(rows: List[Dict[str, Optional[str]]]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for row in rows:
        entity_id = row.get("entity_id")
        comp_id = row.get("comp_id")
        if entity_id and comp_id:
            result[entity_id] = comp_id
    return result


def build_branch_entity_map(rows: List[Dict[str, Optional[str]]]) -> Dict[str, Dict[str, Optional[str]]]:
    result: Dict[str, Dict[str, Optional[str]]] = {}
    for row in rows:
        entity_id = row.get("entity_id")
        if entity_id:
            result[entity_id] = row
    return result


def build_chem_comp_map(rows: List[Dict[str, Optional[str]]]) -> Dict[str, ChemCompInfo]:
    result: Dict[str, ChemCompInfo] = {}
    for row in rows:
        comp_id = row.get("id")
        if not comp_id:
            continue
        result[comp_id] = ChemCompInfo(
            comp_id=comp_id,
            comp_type=row.get("type"),
            name=row.get("name"),
            mon_nstd_flag=row.get("mon_nstd_flag"),
            pdbx_type=row.get("pdbx_type"),
            formula=row.get("formula"),
            formula_weight=row.get("formula_weight"),
        )
    return result


def build_atom_records(rows: List[Dict[str, Optional[str]]]) -> List[AtomRecord]:
    atoms: List[AtomRecord] = []
    for row in rows:
        try:
            x = float(row.get("Cartn_x"))  # type: ignore[arg-type]
            y = float(row.get("Cartn_y"))  # type: ignore[arg-type]
            z = float(row.get("Cartn_z"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue

        atoms.append(
            AtomRecord(
                group_PDB=row.get("group_PDB") or "ATOM",
                atom_id=row.get("id") or "",
                element=row.get("type_symbol") or "",
                atom_name=row.get("label_atom_id") or "",
                alt_id=row.get("label_alt_id") or ".",
                comp_id=row.get("label_comp_id") or "",
                label_asym_id=row.get("label_asym_id") or "",
                label_entity_id=row.get("label_entity_id") or "",
                label_seq_id=row.get("label_seq_id"),
                auth_asym_id=row.get("auth_asym_id") or "",
                auth_seq_id=row.get("auth_seq_id") or "",
                auth_comp_id=row.get("auth_comp_id") or "",
                ins_code=row.get("pdbx_PDB_ins_code") or "",
                x=x,
                y=y,
                z=z,
            )
        )
    return atoms


# -----------------------------------------------------------------------------
# Classification helpers
# -----------------------------------------------------------------------------
def polymer_type_to_class(poly_type: Optional[str]) -> str:
    if not poly_type:
        return "Polymer"
    p = poly_type.lower()
    if "polyribonucleotide" in p and "deoxy" not in p:
        return "RNA"
    if "polydeoxyribonucleotide" in p and "ribo" not in p:
        return "DNA"
    if "polypeptide" in p:
        return "Protein"
    if "hybrid" in p:
        return "Hybrid polymer"
    return "Other polymer"


def classify_nonpoly_subtype(
    comp_id: str,
    chem_comp: Optional[ChemCompInfo],
    entity_description: Optional[str],
) -> str:
    cid = (comp_id or "").upper()
    desc = (entity_description or "").upper()
    cc_name = ((chem_comp.name if chem_comp else "") or "").upper()
    cc_type = ((chem_comp.comp_type if chem_comp else "") or "").upper()
    text = f"{desc} {cc_name} {cc_type}"

    if cid in WATER_COMP_IDS or "WATER" in text:
        return "Water"
    if cid in ION_COMP_IDS or " ION" in text or text.endswith("ION"):
        return "Ion/Inorganic"
    if cid in BUFFER_SOLVENT_COMP_IDS or "BUFFER" in text or "SOLVENT" in text or "TRIS" in text:
        return "Buffer/Solvent"
    return "Ligand"


def classify_branched_subtype(entity_description: Optional[str]) -> str:
    desc = (entity_description or "").lower()
    if any(x in desc for x in ["sucrose", "glucose", "fructose", "mannose", "glycan", "oligosaccharide"]):
        return "Branched carbohydrate"
    return "Branched"


def build_chain_class_map(
    entity_map: Dict[str, EntityInfo],
    polymer_map: Dict[str, PolymerInfo],
    struct_asym_map: Dict[str, str],
    nonpoly_entity_map: Dict[str, str],
    chem_comp_map: Dict[str, ChemCompInfo],
) -> Dict[str, str]:
    chain_class: Dict[str, str] = {}
    for label_asym_id, entity_id in struct_asym_map.items():
        entity = entity_map.get(entity_id)
        if not entity:
            chain_class[label_asym_id] = "Unknown"
            continue

        if entity.entity_type == "polymer":
            poly = polymer_map.get(entity_id)
            chain_class[label_asym_id] = polymer_type_to_class(poly.poly_type if poly else None)
        elif entity.entity_type == "non-polymer":
            comp_id = nonpoly_entity_map.get(entity_id, "")
            chem = chem_comp_map.get(comp_id)
            chain_class[label_asym_id] = classify_nonpoly_subtype(comp_id, chem, entity.description)
        elif entity.entity_type == "branched":
            chain_class[label_asym_id] = classify_branched_subtype(entity.description)
        elif entity.entity_type == "water":
            chain_class[label_asym_id] = "Water"
        else:
            chain_class[label_asym_id] = entity.entity_type

    return chain_class


# -----------------------------------------------------------------------------
# Control / checkpoint helpers
# -----------------------------------------------------------------------------
def build_polymer_component_membership(
    entity_poly_seq_map: Dict[str, List[Dict[str, Optional[str]]]],
    poly_seq_scheme_map: Dict[str, List[Dict[str, Optional[str]]]],
) -> Dict[str, set]:
    result: Dict[str, set] = defaultdict(set)

    for entity_id, rows in entity_poly_seq_map.items():
        for row in rows:
            mon_id = row.get("mon_id")
            if mon_id:
                result[entity_id].add(mon_id)

    for entity_id, rows in poly_seq_scheme_map.items():
        for row in rows:
            mon_id = row.get("mon_id")
            auth_mon_id = row.get("auth_mon_id")
            if mon_id:
                result[entity_id].add(mon_id)
            if auth_mon_id:
                result[entity_id].add(auth_mon_id)

    return dict(result)


def atom_entity_id(atom: AtomRecord, struct_asym_map: Dict[str, str]) -> str:
    return atom.label_entity_id or struct_asym_map.get(atom.label_asym_id, "")


def atom_belongs_to_polymer(
    atom: AtomRecord,
    entity_map: Dict[str, EntityInfo],
    struct_asym_map: Dict[str, str],
) -> bool:
    eid = atom_entity_id(atom, struct_asym_map)
    ent = entity_map.get(eid)
    return bool(ent and ent.entity_type == "polymer")


def polymer_atoms_of_class(
    atoms: List[AtomRecord],
    entity_map: Dict[str, EntityInfo],
    polymer_map: Dict[str, PolymerInfo],
    struct_asym_map: Dict[str, str],
    desired_class: str,
) -> List[AtomRecord]:
    out: List[AtomRecord] = []
    for atom in atoms:
        eid = atom_entity_id(atom, struct_asym_map)
        ent = entity_map.get(eid)
        if not ent or ent.entity_type != "polymer":
            continue
        poly = polymer_map.get(eid)
        if polymer_type_to_class(poly.poly_type if poly else None) == desired_class:
            out.append(atom)
    return out


def branch_atoms(
    atoms: List[AtomRecord],
    entity_map: Dict[str, EntityInfo],
    struct_asym_map: Dict[str, str],
) -> List[AtomRecord]:
    out = []
    for atom in atoms:
        eid = atom_entity_id(atom, struct_asym_map)
        ent = entity_map.get(eid)
        if ent and ent.entity_type == "branched":
            out.append(atom)
    return out


# -----------------------------------------------------------------------------
# Instance enumeration
# -----------------------------------------------------------------------------
def enumerate_polymer_residue_instances(
    entity_map: Dict[str, EntityInfo],
    polymer_map: Dict[str, PolymerInfo],
    entity_poly_seq_map: Dict[str, List[Dict[str, Optional[str]]]],
    poly_seq_scheme_map: Dict[str, List[Dict[str, Optional[str]]]],
    struct_asym_map: Dict[str, str],
    atoms: List[AtomRecord],
) -> List[StructuralInstance]:
    """
    Not used as ligand candidates. Used only as checkpoints / debug / cross-validation.
    """
    instances: List[StructuralInstance] = []
    seen = set()

    for atom in atoms:
        eid = atom_entity_id(atom, struct_asym_map)
        ent = entity_map.get(eid)
        if not ent or ent.entity_type != "polymer":
            continue

        poly = polymer_map.get(eid)
        cat = polymer_type_to_class(poly.poly_type if poly else None)
        key = (eid, atom.label_asym_id, atom.auth_asym_id, atom.auth_seq_id, atom.comp_id)
        if key in seen:
            continue
        seen.add(key)

        instances.append(
            StructuralInstance(
                entity_id=eid,
                entity_kind="polymer",
                category=cat,
                comp_id=atom.comp_id,
                label_asym_id=atom.label_asym_id,
                auth_asym_id=atom.auth_asym_id,
                auth_seq_id=atom.auth_seq_id,
                description=ent.description,
                source="atom_site+entity",
            )
        )
    return instances


def enumerate_nonpoly_instances(
    entity_map: Dict[str, EntityInfo],
    struct_asym_map: Dict[str, str],
    nonpoly_entity_map: Dict[str, str],
    chem_comp_map: Dict[str, ChemCompInfo],
    nonpoly_scheme_rows: List[Dict[str, Optional[str]]],
    atoms: List[AtomRecord],
) -> List[StructuralInstance]:
    instances: List[StructuralInstance] = []

    if nonpoly_scheme_rows:
        for row in nonpoly_scheme_rows:
            entity_id = row.get("entity_id")
            label_asym_id = row.get("asym_id")
            comp_id = row.get("mon_id")
            auth_asym_id = row.get("pdb_strand_id") or row.get("auth_asym_id") or ""
            auth_seq_id = row.get("pdb_seq_num") or row.get("auth_seq_num") or row.get("seq_id") or ""

            if not entity_id or not label_asym_id or not comp_id:
                continue

            ent = entity_map.get(entity_id)
            if not ent or ent.entity_type != "non-polymer":
                continue

            expected_comp = nonpoly_entity_map.get(entity_id)
            warnings: List[str] = []
            if expected_comp and expected_comp != comp_id:
                warnings.append(
                    f"_pdbx_entity_nonpoly comp_id={expected_comp} but _pdbx_nonpoly_scheme mon_id={comp_id}"
                )

            matched = [
                a for a in atoms
                if atom_entity_id(a, struct_asym_map) == entity_id
                and a.label_asym_id == label_asym_id
                and a.comp_id == comp_id
                and (not auth_asym_id or a.auth_asym_id == auth_asym_id)
                and (not auth_seq_id or a.auth_seq_id == auth_seq_id)
            ]
            if not matched:
                matched = [
                    a for a in atoms
                    if atom_entity_id(a, struct_asym_map) == entity_id
                    and a.label_asym_id == label_asym_id
                    and a.comp_id == comp_id
                ]
                if matched:
                    warnings.append("matched non-polymer instance by weaker key")
                    auth_asym_id = auth_asym_id or matched[0].auth_asym_id
                    auth_seq_id = auth_seq_id or matched[0].auth_seq_id
                else:
                    warnings.append("instance in _pdbx_nonpoly_scheme not found in _atom_site")
                    continue

            chem = chem_comp_map.get(comp_id)
            category = classify_nonpoly_subtype(comp_id, chem, ent.description)

            instances.append(
                StructuralInstance(
                    entity_id=entity_id,
                    entity_kind="non-polymer",
                    category=category,
                    comp_id=comp_id,
                    label_asym_id=label_asym_id,
                    auth_asym_id=auth_asym_id or matched[0].auth_asym_id,
                    auth_seq_id=auth_seq_id or matched[0].auth_seq_id,
                    description=ent.description,
                    chem_comp_type=chem.comp_type if chem else None,
                    source="_pdbx_nonpoly_scheme",
                    warnings=warnings,
                )
            )
    else:
        seen = set()
        for atom in atoms:
            eid = atom_entity_id(atom, struct_asym_map)
            ent = entity_map.get(eid)
            if not ent or ent.entity_type != "non-polymer":
                continue

            key = (eid, atom.label_asym_id, atom.auth_asym_id, atom.auth_seq_id, atom.comp_id)
            if key in seen:
                continue
            seen.add(key)

            chem = chem_comp_map.get(atom.comp_id)
            category = classify_nonpoly_subtype(atom.comp_id, chem, ent.description)
            warnings = ["_pdbx_nonpoly_scheme missing; inferred from _atom_site"]

            instances.append(
                StructuralInstance(
                    entity_id=eid,
                    entity_kind="non-polymer",
                    category=category,
                    comp_id=atom.comp_id,
                    label_asym_id=atom.label_asym_id,
                    auth_asym_id=atom.auth_asym_id,
                    auth_seq_id=atom.auth_seq_id,
                    description=ent.description,
                    chem_comp_type=chem.comp_type if chem else None,
                    source="atom_site_fallback",
                    warnings=warnings,
                )
            )

    dedup: Dict[Tuple[str, str, str, str, str], StructuralInstance] = {}
    for inst in instances:
        key = (inst.entity_id, inst.comp_id, inst.label_asym_id, inst.auth_asym_id, inst.auth_seq_id)
        if key not in dedup:
            dedup[key] = inst
        else:
            for w in inst.warnings:
                if w not in dedup[key].warnings:
                    dedup[key].warnings.append(w)

    return list(dedup.values())


def enumerate_branched_instances(
    entity_map: Dict[str, EntityInfo],
    struct_asym_map: Dict[str, str],
    branch_entity_map: Dict[str, Dict[str, Optional[str]]],
    branch_scheme_rows: List[Dict[str, Optional[str]]],
    atoms: List[AtomRecord],
) -> List[StructuralInstance]:
    instances: List[StructuralInstance] = []

    if branch_scheme_rows:
        seen = set()
        for row in branch_scheme_rows:
            entity_id = row.get("entity_id")
            label_asym_id = row.get("asym_id")
            comp_id = row.get("mon_id") or ""
            auth_asym_id = row.get("pdb_asym_id") or row.get("pdb_strand_id") or ""
            auth_seq_id = row.get("pdb_seq_num") or row.get("num") or ""

            if not entity_id or not label_asym_id:
                continue

            ent = entity_map.get(entity_id)
            if not ent or ent.entity_type != "branched":
                continue

            key = (entity_id, label_asym_id, auth_asym_id, auth_seq_id, comp_id)
            if key in seen:
                continue
            seen.add(key)

            instances.append(
                StructuralInstance(
                    entity_id=entity_id,
                    entity_kind="branched",
                    category=classify_branched_subtype(ent.description),
                    comp_id=comp_id,
                    label_asym_id=label_asym_id,
                    auth_asym_id=auth_asym_id,
                    auth_seq_id=auth_seq_id,
                    description=ent.description,
                    source="_pdbx_branch_scheme",
                )
            )
    else:
        seen = set()
        for atom in atoms:
            eid = atom_entity_id(atom, struct_asym_map)
            ent = entity_map.get(eid)
            if not ent or ent.entity_type != "branched":
                continue

            key = (eid, atom.label_asym_id, atom.auth_asym_id, atom.auth_seq_id, atom.comp_id)
            if key in seen:
                continue
            seen.add(key)

            instances.append(
                StructuralInstance(
                    entity_id=eid,
                    entity_kind="branched",
                    category=classify_branched_subtype(ent.description),
                    comp_id=atom.comp_id,
                    label_asym_id=atom.label_asym_id,
                    auth_asym_id=atom.auth_asym_id,
                    auth_seq_id=atom.auth_seq_id,
                    description=ent.description,
                    source="atom_site_fallback",
                    warnings=["_pdbx_branch_scheme missing; inferred from _atom_site"],
                )
            )

    return instances


def get_instance_atoms(
    atoms: List[AtomRecord],
    instance: StructuralInstance,
    struct_asym_map: Dict[str, str],
    alt_loc: Optional[str] = None,
) -> List[AtomRecord]:
    candidates: List[AtomRecord] = []
    for atom in atoms:
        if atom_entity_id(atom, struct_asym_map) != instance.entity_id:
            continue
        if atom.label_asym_id != instance.label_asym_id:
            continue
        if instance.comp_id and atom.comp_id != instance.comp_id:
            continue
        if instance.auth_asym_id and atom.auth_asym_id != instance.auth_asym_id:
            continue
        if instance.auth_seq_id and atom.auth_seq_id != instance.auth_seq_id:
            continue
        if atom.element.upper() == "H":
            continue
        candidates.append(atom)

    if not candidates:
        return []

    plain = [a for a in candidates if a.alt_id in {"", ".", "?", None}]
    alts = [a for a in candidates if a.alt_id not in {"", ".", "?", None}]

    if not alts:
        return plain

    if alt_loc is None:
        counts: Dict[str, int] = defaultdict(int)
        for a in alts:
            counts[a.alt_id] += 1
        chosen = max(counts, key=counts.get)
    else:
        chosen = alt_loc

    return plain + [a for a in alts if a.alt_id == chosen]


# -----------------------------------------------------------------------------
# Contact detection
# -----------------------------------------------------------------------------
def find_contacts(
    source_atoms: List[AtomRecord],
    target_atoms: List[AtomRecord],
    threshold: float = 5.0,
) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, str, str], Dict[str, Any]]]:
    src = [a for a in source_atoms if a.element.upper() != "H"]
    tgt = [a for a in target_atoms if a.element.upper() != "H"]

    if not src or not tgt:
        return [], {}

    src_coords = np.array([[a.x, a.y, a.z] for a in src], dtype=float)
    tgt_coords = np.array([[a.x, a.y, a.z] for a in tgt], dtype=float)
    tree = cKDTree(tgt_coords)

    contacts: List[Dict[str, Any]] = []
    for i, sa in enumerate(src):
        nearby = tree.query_ball_point(src_coords[i], threshold)
        for j in nearby:
            ta = tgt[j]
            d = float(np.linalg.norm(src_coords[i] - tgt_coords[j]))
            if d <= threshold:
                contacts.append({
                    "src_atom": sa.atom_name,
                    "src_elem": sa.element,
                    "tgt_chain": ta.auth_asym_id,
                    "tgt_label_asym_id": ta.label_asym_id,
                    "tgt_resname": ta.comp_id,
                    "tgt_resseq": ta.auth_seq_id,
                    "tgt_atom": ta.atom_name,
                    "tgt_elem": ta.element,
                    "distance": d,
                })

    by_res: Dict[Tuple[str, str, str], Dict[str, Any]] = defaultdict(
        lambda: {"min_d": 1e9, "n_contacts": 0, "atoms": []}
    )
    for c in contacts:
        key = (str(c["tgt_chain"]), str(c["tgt_resname"]), str(c["tgt_resseq"]))
        by_res[key]["n_contacts"] += 1
        by_res[key]["atoms"].append(f"{c['src_atom']}···{c['tgt_atom']}")
        if float(c["distance"]) < float(by_res[key]["min_d"]):
            by_res[key]["min_d"] = float(c["distance"])

    return contacts, dict(by_res)


def chains_in_contact(contacts: List[Dict[str, Any]]) -> List[str]:
    return sorted({str(c["tgt_chain"]) for c in contacts if c.get("tgt_chain")})


def print_residue_table(by_res: Dict[Tuple[str, str, str], Dict[str, Any]], max_atoms: int = 5) -> None:
    print(
        f"    {'Chain':<6}{'Res':<8}{'#':<8}{'min_d (Å)':<12}"
        f"{'#contacts':<10}{'atoms (src···tgt)'}"
    )
    for (chain_id, resname, resseq), info in sorted(by_res.items(), key=lambda x: (x[0][0], float(x[1]["min_d"]))):
        atoms = ", ".join(info["atoms"][:max_atoms])
        if len(info["atoms"]) > max_atoms:
            atoms += f", … (+{len(info['atoms']) - max_atoms})"
        print(
            f"    {chain_id:<6}{resname:<8}{resseq:<8}{float(info['min_d']):<12.2f}"
            f"{int(info['n_contacts']):<10}{atoms}"
        )


def make_output_row(
    pdb_id: str,
    instance: StructuralInstance,
    contact: Dict[str, Any],
    target_type: str,
) -> Dict[str, Any]:
    return {
        "PDB_ID": pdb_id,
        "Entity_ID": instance.entity_id,
        "Entity_Kind": instance.entity_kind,
        "Comp_ID": instance.comp_id,
        "Instance_Category": instance.category,
        "Label_Asym_ID": instance.label_asym_id,
        "Auth_Chain": instance.auth_asym_id,
        "Auth_Seq_ID": instance.auth_seq_id,
        "ChemComp_Type": instance.chem_comp_type or "",
        "Target_Type": target_type,
        "Src_Atom": contact["src_atom"],
        "Src_Elem": contact["src_elem"],
        "Tgt_Chain": contact["tgt_chain"],
        "Tgt_Resname": contact["tgt_resname"],
        "Tgt_ResSeq": contact["tgt_resseq"],
        "Tgt_Atom": contact["tgt_atom"],
        "Tgt_Elem": contact["tgt_elem"],
        "Distance_A": round(float(contact["distance"]), 3),
    }


# -----------------------------------------------------------------------------
# Reporting helpers
# -----------------------------------------------------------------------------
def print_checkpoint_report(
    entity_map: Dict[str, EntityInfo],
    polymer_map: Dict[str, PolymerInfo],
    entity_poly_seq_map: Dict[str, List[Dict[str, Optional[str]]]],
    poly_seq_scheme_map: Dict[str, List[Dict[str, Optional[str]]]],
    struct_asym_map: Dict[str, str],
    nonpoly_entity_map: Dict[str, str],
    branch_entity_map: Dict[str, Dict[str, Optional[str]]],
    chem_comp_map: Dict[str, ChemCompInfo],
    atoms: List[AtomRecord],
) -> None:
    n_poly = sum(1 for e in entity_map.values() if e.entity_type == "polymer")
    n_nonpoly = sum(1 for e in entity_map.values() if e.entity_type == "non-polymer")
    n_branched = sum(1 for e in entity_map.values() if e.entity_type == "branched")
    n_water = sum(1 for e in entity_map.values() if e.entity_type == "water")

    print("\n  Control checkpoints:")
    print(f"    entity.type polymer:          {n_poly}")
    print(f"    entity.type non-polymer:      {n_nonpoly}")
    print(f"    entity.type branched:         {n_branched}")
    print(f"    entity.type water:            {n_water}")
    print(f"    _entity_poly rows/entities:   {len(polymer_map)}")
    print(f"    _entity_poly_seq entities:    {len(entity_poly_seq_map)}")
    print(f"    _pdbx_poly_seq_scheme ents:   {len(poly_seq_scheme_map)}")
    print(f"    _struct_asym mappings:        {len(struct_asym_map)}")
    print(f"    _pdbx_entity_nonpoly ents:    {len(nonpoly_entity_map)}")
    print(f"    _pdbx_entity_branch ents:     {len(branch_entity_map)}")
    print(f"    _chem_comp entries:           {len(chem_comp_map)}")
    print(f"    _atom_site atoms:             {len(atoms)}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Entity-aware RNA / ligand contact checker for mmCIF files. "
            "Uses a field-priority checklist with fallbacks and control checkpoints."
        )
    )
    parser.add_argument("pdb_id", help="PDB ID, for example 9E5T")
    parser.add_argument("-f", "--cif-file", default=None, help="Use a local mmCIF file")
    parser.add_argument("-t", "--threshold", type=float, default=5.0, help="Distance cutoff in Å")
    parser.add_argument("-o", "--output", default=None, help="Write detailed TSV")
    parser.add_argument("--cache-dir", default=".", help="CIF cache directory")
    parser.add_argument("--show-all", action="store_true", help="Also print discarded instances")
    parser.add_argument("--include-ions", action="store_true", help="Keep Ion/Inorganic instances")
    parser.add_argument("--include-buffers", action="store_true", help="Keep Buffer/Solvent instances")
    parser.add_argument("--include-branched", action="store_true", help="Keep branched entities if they contact RNA")
    parser.add_argument("--comp-id", default=None, help="Restrict to a component ID")
    parser.add_argument("--label-asym-id", default=None, help="Restrict to a label_asym_id")
    parser.add_argument("--auth-chain", default=None, help="Restrict to an auth chain")
    args = parser.parse_args()

    pdb_id = args.pdb_id.upper()

    print("=" * 100)
    print(f"  Entity-aware contact check: {pdb_id}")
    print(f"  Threshold: {args.threshold} Å (heavy atoms only)")
    print("  Checklist sources: _entity, _entity_poly, _entity_poly_seq, _pdbx_poly_seq_scheme,")
    print("                     _struct_asym, _pdbx_entity_nonpoly, _pdbx_nonpoly_scheme,")
    print("                     _pdbx_entity_branch, _pdbx_branch_scheme, _atom_site, _chem_comp")
    print("=" * 100)

    cif_path = args.cif_file or fetch_cif(pdb_id, args.cache_dir)
    if not cif_path or not os.path.exists(cif_path):
        print(f"  ERROR: could not obtain CIF for {pdb_id}")
        sys.exit(1)

    print("\n[1/4] Reading CIF metadata...")
    tables = load_tables(cif_path)

    entity_map = build_entity_map(tables["entity"])
    polymer_map = build_polymer_map(tables["entity_poly"])
    entity_poly_seq_map = build_entity_poly_seq_map(tables["entity_poly_seq"])
    poly_seq_scheme_map = build_poly_seq_scheme_map(tables["pdbx_poly_seq_scheme"])
    struct_asym_map = build_struct_asym_map(tables["struct_asym"])
    nonpoly_entity_map = build_nonpoly_entity_map(tables["entity_nonpoly"])
    branch_entity_map = build_branch_entity_map(tables["entity_branch"])
    chem_comp_map = build_chem_comp_map(tables["chem_comp"])
    atoms = build_atom_records(tables["atom_site"])

    print(f"  Entities:              {len(entity_map)}")
    print(f"  Polymer entities:      {len(polymer_map)}")
    print(f"  Polymer seq entities:  {len(entity_poly_seq_map)}")
    print(f"  Poly seq-scheme ents:  {len(poly_seq_scheme_map)}")
    print(f"  Struct asym mappings:  {len(struct_asym_map)}")
    print(f"  Non-poly entities:     {len(nonpoly_entity_map)}")
    print(f"  Branched entities:     {len(branch_entity_map)}")
    print(f"  Chem comps:            {len(chem_comp_map)}")
    print(f"  Atom records:          {len(atoms)}")

    print_checkpoint_report(
        entity_map, polymer_map, entity_poly_seq_map, poly_seq_scheme_map,
        struct_asym_map, nonpoly_entity_map, branch_entity_map, chem_comp_map, atoms
    )

    print("\n[2/4] Building polymer and non-polymer classification...")
    chain_class_map = build_chain_class_map(
        entity_map=entity_map,
        polymer_map=polymer_map,
        struct_asym_map=struct_asym_map,
        nonpoly_entity_map=nonpoly_entity_map,
        chem_comp_map=chem_comp_map,
    )

    rna_label_asym = sorted([k for k, v in chain_class_map.items() if v == "RNA"])
    dna_label_asym = sorted([k for k, v in chain_class_map.items() if v == "DNA"])
    prot_label_asym = sorted([k for k, v in chain_class_map.items() if v == "Protein"])
    branch_label_asym = sorted([k for k, v in chain_class_map.items() if "Branched" in v])

    print(f"  RNA label_asym_id:     {rna_label_asym if rna_label_asym else 'none'}")
    print(f"  DNA label_asym_id:     {dna_label_asym if dna_label_asym else 'none'}")
    print(f"  Protein label_asym_id: {prot_label_asym if prot_label_asym else 'none'}")
    print(f"  Branched asym_id:      {branch_label_asym if branch_label_asym else 'none'}")

    if not rna_label_asym:
        print("\n  No RNA polymer chains found from correlated entity metadata.")
        if args.output:
            cols = [
                "PDB_ID", "Entity_ID", "Entity_Kind", "Comp_ID", "Instance_Category",
                "Label_Asym_ID", "Auth_Chain", "Auth_Seq_ID", "ChemComp_Type",
                "Target_Type", "Src_Atom", "Src_Elem", "Tgt_Chain", "Tgt_Resname",
                "Tgt_ResSeq", "Tgt_Atom", "Tgt_Elem", "Distance_A"
            ]
            with open(args.output, "w", encoding="utf-8") as handle:
                handle.write("\t".join(cols) + "\n")
            print(f"  Wrote empty TSV header to: {args.output}")
        sys.exit(0)

    print("\n[3/4] Enumerating structural instances...")
    polymer_instances = enumerate_polymer_residue_instances(
        entity_map=entity_map,
        polymer_map=polymer_map,
        entity_poly_seq_map=entity_poly_seq_map,
        poly_seq_scheme_map=poly_seq_scheme_map,
        struct_asym_map=struct_asym_map,
        atoms=atoms,
    )
    nonpoly_instances = enumerate_nonpoly_instances(
        entity_map=entity_map,
        struct_asym_map=struct_asym_map,
        nonpoly_entity_map=nonpoly_entity_map,
        chem_comp_map=chem_comp_map,
        nonpoly_scheme_rows=tables["nonpoly_scheme"],
        atoms=atoms,
    )
    branched_instances = enumerate_branched_instances(
        entity_map=entity_map,
        struct_asym_map=struct_asym_map,
        branch_entity_map=branch_entity_map,
        branch_scheme_rows=tables["branch_scheme"],
        atoms=atoms,
    )

    instances = nonpoly_instances + branched_instances

    if args.comp_id:
        instances = [inst for inst in instances if inst.comp_id.upper() == args.comp_id.upper()]
    if args.label_asym_id:
        instances = [inst for inst in instances if inst.label_asym_id == args.label_asym_id]
    if args.auth_chain:
        instances = [inst for inst in instances if inst.auth_asym_id == args.auth_chain]

    print(f"  Polymer residue instances: {len(polymer_instances)}")
    print(f"  Non-polymer instances:     {len(nonpoly_instances)}")
    print(f"  Branched instances:        {len(branched_instances)}")
    print(f"  Candidate source objects:  {len(instances)}")

    for inst in instances:
        warn = " [warn]" if inst.warnings else ""
        print(
            f"    {inst.entity_kind:<11} entity={inst.entity_id:<4} comp={inst.comp_id or '-':<8} "
            f"cat={inst.category:<22} label_asym={inst.label_asym_id:<6} "
            f"auth_chain={inst.auth_asym_id:<6} auth_seq={inst.auth_seq_id:<6}{warn}"
        )

    rna_atoms = polymer_atoms_of_class(atoms, entity_map, polymer_map, struct_asym_map, "RNA")
    dna_atoms = polymer_atoms_of_class(atoms, entity_map, polymer_map, struct_asym_map, "DNA")
    prot_atoms = polymer_atoms_of_class(atoms, entity_map, polymer_map, struct_asym_map, "Protein")

    print("\n[4/4] Computing contacts...\n")

    kept = 0
    discarded = 0
    output_rows: List[Dict[str, Any]] = []

    for inst in instances:
        if inst.entity_kind == "branched":
            keepable = args.include_branched
        elif inst.category == "Water":
            keepable = False
        elif inst.category == "Ion/Inorganic":
            keepable = args.include_ions
        elif inst.category == "Buffer/Solvent":
            keepable = args.include_buffers
        else:
            keepable = True

        src_atoms = get_instance_atoms(atoms, inst, struct_asym_map)
        if not src_atoms:
            discarded += 1
            if args.show_all:
                print("─" * 100)
                print(
                    f" {inst.comp_id or '-'} entity={inst.entity_id} cat={inst.category} "
                    f"label_asym={inst.label_asym_id} auth_chain={inst.auth_asym_id} "
                    f"auth_seq={inst.auth_seq_id} [DISCARDED: no atoms matched]"
                )
            continue

        rna_contacts, rna_res = find_contacts(src_atoms, rna_atoms, args.threshold)
        prot_contacts, prot_res = find_contacts(src_atoms, prot_atoms, args.threshold) if prot_atoms else ([], {})
        dna_contacts, dna_res = find_contacts(src_atoms, dna_atoms, args.threshold) if dna_atoms else ([], {})

        reasons = []
        if not keepable:
            reasons.append(f"category={inst.category} excluded by policy")
        if not rna_contacts:
            reasons.append("no RNA contact")

        if reasons:
            discarded += 1
            if args.show_all:
                print("─" * 100)
                print(
                    f" {inst.comp_id or '-'} entity={inst.entity_id} kind={inst.entity_kind} "
                    f"cat={inst.category} label_asym={inst.label_asym_id} "
                    f"auth_chain={inst.auth_asym_id} auth_seq={inst.auth_seq_id} "
                    f"[DISCARDED: {', '.join(reasons)}]"
                )
                for w in inst.warnings:
                    print(f"    warning: {w}")
                if prot_contacts:
                    print(f"    protein contacts: {len(prot_contacts)}")
                if dna_contacts:
                    print(f"    DNA contacts:     {len(dna_contacts)}")
                print()
            continue

        kept += 1
        rna_min = min(float(c["distance"]) for c in rna_contacts)
        rna_hit_chains = chains_in_contact(rna_contacts)

        print("─" * 100)
        print(
            f" {inst.comp_id or '-'} entity={inst.entity_id} kind={inst.entity_kind} "
            f"cat={inst.category} label_asym={inst.label_asym_id} "
            f"auth_chain={inst.auth_asym_id} auth_seq={inst.auth_seq_id} [KEPT]"
        )
        print(f"  Source heavy atoms: {len(src_atoms)}")
        if inst.chem_comp_type:
            print(f"  chem_comp.type:     {inst.chem_comp_type}")
        if inst.description:
            print(f"  entity description: {inst.description}")
        print(f"  source mapping:     {inst.source}")
        for w in inst.warnings:
            print(f"  warning:            {w}")

        print(
            f"\n  RNA contacts:       {len(rna_contacts)} atom pairs · "
            f"{len(rna_res)} residues · closest = {rna_min:.2f} Å · "
            f"chain(s): {', '.join(rna_hit_chains)}"
        )
        print()
        print_residue_table(rna_res)

        if prot_contacts:
            p_min = min(float(c["distance"]) for c in prot_contacts)
            p_hit = chains_in_contact(prot_contacts)
            print(
                f"\n  Protein contacts:   {len(prot_contacts)} atom pairs · "
                f"{len(prot_res)} residues · closest = {p_min:.2f} Å · "
                f"chain(s): {', '.join(p_hit)}"
            )
            print()
            print_residue_table(prot_res)

        if dna_contacts:
            d_min = min(float(c["distance"]) for c in dna_contacts)
            d_hit = chains_in_contact(dna_contacts)
            print(
                f"\n  DNA contacts:       {len(dna_contacts)} atom pairs · "
                f"{len(dna_res)} residues · closest = {d_min:.2f} Å · "
                f"chain(s): {', '.join(d_hit)}"
            )
            print()
            print_residue_table(dna_res)

        for c in rna_contacts:
            output_rows.append(make_output_row(pdb_id, inst, c, "RNA"))
        for c in prot_contacts:
            output_rows.append(make_output_row(pdb_id, inst, c, "Protein"))
        for c in dna_contacts:
            output_rows.append(make_output_row(pdb_id, inst, c, "DNA"))

        print()

    print("─" * 100)
    print(f"  Total candidate instances examined: {len(instances)}")
    print(f"  Kept:                              {kept}")
    print(f"  Discarded:                         {discarded}")
    print("=" * 100)

    if args.output:
        cols = [
            "PDB_ID", "Entity_ID", "Entity_Kind", "Comp_ID", "Instance_Category",
            "Label_Asym_ID", "Auth_Chain", "Auth_Seq_ID", "ChemComp_Type",
            "Target_Type", "Src_Atom", "Src_Elem", "Tgt_Chain", "Tgt_Resname",
            "Tgt_ResSeq", "Tgt_Atom", "Tgt_Elem", "Distance_A"
        ]
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write("\t".join(cols) + "\n")
            for row in output_rows:
                handle.write("\t".join(str(row[col]) for col in cols) + "\n")
        print(f"  Detailed table: {args.output} ({len(output_rows)} rows)")

    print("  Done.")


if __name__ == "__main__":
    main()
