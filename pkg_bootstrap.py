import os
import json
import shutil


def _add_dir_to_pkg(exporter, dir_path, package_root, manifest):
    base_name = os.path.basename(os.path.normpath(dir_path))
    for root, _, files in os.walk(dir_path):
        rel_dir = os.path.relpath(root, dir_path)
        pkg = package_root + '.' + base_name
        if rel_dir != '.':
            pkg += '.' + rel_dir.replace(os.sep, '.')
        for fname in files:
            src = os.path.join(root, fname)
            # Save file content as a binary resource inside the package
            with open(src, 'rb') as fsrc:
                data = fsrc.read()
            exporter.save_binary(pkg, fname, data)
            manifest.append({
                "original_dir": base_name,
                "relative_path": (fname if rel_dir == '.' else os.path.join(rel_dir, fname)).replace('\\', '/'),
                "package": pkg,
                "resource": fname
            })


def embed_model_dirs(exporter, project_root_dir):
    """
    Embed model directories (hc_model, skin_type_model) into the package and
    write a manifest under the logical package "assets".
    """
    manifest = []
    hc_dir = os.path.join(project_root_dir, "hc_model")
    st_dir = os.path.join(project_root_dir, "skin_type_model")
    _add_dir_to_pkg(exporter, hc_dir, "assets", manifest)
    _add_dir_to_pkg(exporter, st_dir, "assets", manifest)
    exporter.save_text("assets", "manifest.json", json.dumps(manifest))


def extract_assets(importer, dest_root_dir):
    """
    Extract packaged directories (hc_model, skin_type_model) into dest_root_dir
    if they are missing. Uses the assets/manifest.json included in the package.
    """
    hc_dir = os.path.join(dest_root_dir, "hc_model")
    st_dir = os.path.join(dest_root_dir, "skin_type_model")

    manifest_text = importer.load_text("assets", "manifest.json")
    manifest = json.loads(manifest_text)

    def _extract_dir(base_name, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        entries = [e for e in manifest if e.get("original_dir") == base_name]
        for e in entries:
            target_path = os.path.join(out_dir, e["relative_path"].replace('/', os.sep))
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            reader = importer.get_resource_reader(e["package"])  # stream to avoid loading entire file in memory
            with reader.open_resource(e["resource"]) as src, open(target_path, 'wb') as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)

    # Only extract if missing a key file
    if not os.path.exists(os.path.join(hc_dir, "config.json")):
        _extract_dir("hc_model", hc_dir)
    if not os.path.exists(os.path.join(st_dir, "config.json")):
        _extract_dir("skin_type_model", st_dir)


