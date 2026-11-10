"""Fail-fast startup and scenario-scoped cleanup for sequential experiments."""


def start_participant(api, **configuration):
    created = api.create_container(**configuration)
    identifier = created["Id"]
    try:
        api.start(identifier)
    except Exception:
        api.remove_container(identifier, force=True)
        raise
    return identifier


def stop_scenario_participants(client, prefix, scenario_name):
    if not scenario_name or "/" in scenario_name or "\\" in scenario_name:
        raise ValueError("An explicit scenario name is required for participant cleanup")
    config_path = f"/nebula/app/config/{scenario_name}/"
    for container in client.containers.list(all=True):
        if not container.name.startswith(prefix):
            continue
        labels = container.attrs.get("Config", {}).get("Labels") or {}
        label = labels.get("nebula.scenario")
        command = container.attrs.get("Config", {}).get("Cmd") or []
        belongs = label == scenario_name if label else any(config_path in token for token in command)
        if belongs:
            try:
                container.remove(force=True)
            except Exception as error:
                if getattr(error, "status_code", None) != 404:
                    raise


def remove_empty_network(client, network_name):
    try:
        network = client.networks.get(network_name)
        if not network.attrs.get("Containers"):
            network.remove()
    except Exception as error:
        # Concurrent cleanup can remove it first, or another scenario can attach.
        if getattr(error, "status_code", None) not in (404, 409):
            raise
