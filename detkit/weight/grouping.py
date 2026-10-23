"""Group repeated photographs by physical sample, never by photograph."""
import json


def pile_key(sample):
    """Use globally unique pile IDs (e.g. A001/J001); old data falls back to file."""
    return json.dumps([sample.crop or "default",
                       str(sample.meta.get("pile_id") or f"file:{sample.file}")])


def group_samples(samples):
    groups = {}
    for sample in samples:
        groups.setdefault(pile_key(sample), []).append(sample)
    return groups
