# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT
"""Build paired fixed-look evaluation data locally, without changing sources.

Videos are retained as source references, not newly rendered v2 previews.
The destination must not exist. Publication/replacement is a separate operation.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil


def read_config(row):
    value = row['environment_config']
    return json.loads(value) if isinstance(value, str) else deepcopy(value)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scene_with_look(source, template):
    result = deepcopy(template)
    for key in ('table', 'table2'):
        if key in source:
            result[key] = deepcopy(source[key])
            if result[key] is not None:
                result[key]['material'] = deepcopy(template['table']['material'])
        else:
            result.pop(key, None)
    return result


def geometry(config):
    actors = {}
    for key, value in config['layout']['actors'].items():
        if key.startswith('distractor_'):
            continue
        actors[key] = {k: v for k, v in value.items() if k not in ('material', 'shader_params')}
    return {'spatial': {k: v for k, v in config['dr_state_dict']['spatial'].items()
                        if k in ('g1_sonic', 'g1_wholebody', config['dr_state_dict']['target']['uid'])},
            'actors': actors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--look', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if str(args.output.resolve()).startswith('/horizon-bucket/'):
        raise ValueError('This builder only writes local datasets')
    report = []
    for task in ('BottleBin', 'BowlSink', 'CoffeeCart'):
        name = f'G1Wholebody{task}Teleop-v0'
        source = args.source / name / 'dr-level-3'
        look = args.look / name / 'dr-level-3'
        template = read_config(json.loads((look / 'meta/episodes.jsonl').read_text().splitlines()[0]))
        rows = [json.loads(line) for line in (source / 'meta/episodes.jsonl').read_text().splitlines()]
        assert len(rows) == 30
        destination = args.output / name / 'dr-level-3'
        shutil.copytree(source, destination)
        transformed = []
        for row in rows:
            original = read_config(row)
            config = deepcopy(original)
            state = config['dr_state_dict']
            target = state['target']['uid']
            state['spatial'] = {k: v for k, v in state['spatial'].items()
                                if k in ('g1_sonic', 'g1_wholebody', target)}
            state['distractors'] = {}
            config['dr_cfgs']['distractors'].update(number_of_distractors=0, include=None)
            state['scene'] = scene_with_look(state['scene'], template['dr_state_dict']['scene'])
            state['lighting'] = deepcopy(template['dr_state_dict']['lighting'])
            state['material'] = deepcopy(template['dr_state_dict']['material'])
            layout = config['layout']
            layout['actors'] = {k: v for k, v in layout['actors'].items() if not k.startswith('distractor_')}
            layout['scene'] = scene_with_look(layout['scene'], template['layout']['scene'])
            layout['lights'] = deepcopy(template['layout']['lights'])
            for key, actor in layout['actors'].items():
                if key in template['layout']['actors'] and 'material' in template['layout']['actors'][key]:
                    actor['material'] = deepcopy(template['layout']['actors'][key]['material'])
            assert geometry(config) == geometry(original), (task, row['episode_index'])
            for key in ('table', 'table2'):
                before = original['dr_state_dict']['scene'].get(key)
                after = state['scene'].get(key)
                if before is not None:
                    assert before['pose'] == after['pose'] and before['size'] == after['size']
            new_row = deepcopy(row)
            new_row['environment_config'] = json.dumps(config) if isinstance(row['environment_config'], str) else config
            transformed.append(new_row)
        (destination / 'meta/episodes.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in transformed))
        provenance = {
            'version': 2, 'source': str(source.resolve()), 'fixed_look_source': str(look.resolve()),
            'episode_mapping': 'identity: v2 episode i uses 3real episode i',
            'distractors': 0, 'runtime_dr_override': None,
            'robot_runtime': 'g1comp-pan0-tilt10-v1',
            'videos': 'Unmodified 3real reference videos; NOT renders of the v2 environment.',
            'joint_state': 'Source parquet preserved; runtime reset/warmup semantics unchanged.',
        }
        (destination / 'meta/scene_env.json').write_text(json.dumps(provenance, indent=2) + '\n')
        unchanged = [f for f in source.rglob('*') if f.is_file() and f.relative_to(source).parts[0] in ('data', 'videos')]
        assert all(digest(f) == digest(destination / f.relative_to(source)) for f in unchanged)
        configs = [read_config(r) for r in transformed]
        assert len({json.dumps(c['dr_state_dict']['spatial'], sort_keys=True) for c in configs}) == 30
        assert len({json.dumps(c['dr_state_dict']['lighting'], sort_keys=True) for c in configs}) == 1
        assert len({json.dumps(c['dr_state_dict']['material'], sort_keys=True) for c in configs}) == 1
        report.append({'task': name, 'episodes': 30, 'unique_spatial_states': 30,
                       'geometry_matches_source': True, 'data_video_hashes_match_source': True,
                       'source_metadata_sha256': digest(source / 'meta/episodes.jsonl')})
    (args.output / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    (args.output / 'README.md').write_text(
        '# Same-background evaluation v2\n\n'
        'Each task contains 30 distinct 3real initial spatial states, paired by episode index.\n'
        'Robot/task geometry and parquet states are preserved. Distractors are removed.\n'
        'Room transform, materials and lights use the original samebg fixed-look template.\n'
        'Use g1comp pan0/tilt10 and restore stored states without a runtime DR override.\n\n'
        'Videos are unchanged 3real reference clips, NOT new v2 previews. No rendering was performed.\n'
        'This data conversion does not change runtime joint restoration or warmup behavior.\n'
        'validation.json records data-level checks; simulator validation is separate.\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
