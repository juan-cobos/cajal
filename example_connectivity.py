from pathlib import Path

import numpy as np

from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache

output_dir = Path("data")
output_dir.mkdir(exist_ok=True)
manifest_file = output_dir / "manifest.json"

mcc = MouseConnectivityCache(manifest_file=manifest_file)
structure_tree = mcc.get_structure_tree()

input_regions = ["VTA", "DR"]
output_regions = ["ACB", "VISp", "MOp"]

annotation, _ = mcc.get_annotation_volume()
resolution = 25

region_info = {}
for acronym in set(input_regions + output_regions):
    struct = structure_tree.get_structures_by_acronym([acronym])[0]
    descendant_ids = structure_tree.descendant_ids([struct["id"]])[0]
    coords = np.argwhere(np.isin(annotation, descendant_ids))
    centroid = (coords.mean(axis=0) * resolution).tolist()
    region_info[acronym] = {"id": struct["id"], "centroid": centroid}

print("Region centroids (AP, DV, LR in microns):")
for acronym, info in region_info.items():
    print(f"  {acronym} (id={info['id']}): {info['centroid']}")

output_region_ids = [region_info[a]["id"] for a in output_regions]

for input_acronym in input_regions:
    input_id = region_info[input_acronym]["id"]
    experiments = mcc.get_experiments(cre=False, injection_structure_ids=[input_id])
    print(f"\n{input_acronym}: {len(experiments)} experiments")

    if not len(experiments):
        print(f"  No wild-type experiments found for {input_acronym}, skipping")
        continue

    unionizes = mcc.get_structure_unionizes(
        [e["id"] for e in experiments],
        is_injection=False,
        structure_ids=output_region_ids,
        include_descendants=True,
    )

    for output_acronym in output_regions:
        subset = unionizes[unionizes["structure_id"] == region_info[output_acronym]["id"]]
        if len(subset) > 0:
            print(
                f"  {input_acronym} -> {output_acronym}: "
                f"mean projection_density={subset['projection_density'].mean():.6f}, "
                f"mean projection_intensity={subset['projection_intensity'].mean():.4f}"
            )
