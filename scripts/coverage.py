
import os
import argparse

from inspect import getmembers, isbuiltin, isclass, isfunction, ismodule
from types import ModuleType
from typing import Dict, List, Set


class PublicInterfaceFinder():
    """A cool finder that can help you find (nearly) all public API under specific Python module.

    Example:

        >>> import megengine  # Or any other packages you like. XD
        >>> finder = PublicInterfaceFinder(megengine)
        >>> api_info = finder.get_api_info()

        All APIs like ``megengine.functional.add`` would be found and storaged in ``api_info`` dict.
        Exclude Python builtin objects and those imported from other packages. See some excluded cases:

        * ``megengine.module.module.ABCMeta`` -- ``ABCMeta`` class is from ``abc`` actually;
        * ``megengine.traced_module.traced_module.isclass`` -- ``isclass`` function is from ``inspect``;
        * ``numpy`` / ``xml`` -- those modules might be imported in submodules of ``megengine``.

    Note:

        Different method, function, class objects... may have the same ``id``, such as:

        * ``megengine.module.module.Module``
        * ``megengine.traced_module.module_tracer.Module``
        * ``megengine.module.elemwise.Module`` ... so many of alias.

        So we choose an object's ``id()`` as the dict key making it easier to bind API with its brothers.
        Calling ``api_info[id]`` will return a set of different API names with same id.

        Corner Case:

        * #1 megengine.module.traced_module.Module
        * #2 from megengine.data.transform.vision import _functional as F

    Warning:

        Only function and class interface would be recorded in ``api_info`` by default.
        Methods of class are not recorded beacuse it's a piece of cake to find their methods.

    Args:
        module: the top module where to start searching and recording public interfaces.
        ingore_keywords: If the API contains any keyword in the list, it would not be recored.
            But notice that an alias could has the totally different name and not be ignored.

    Methods:
        get_api_info: get the built ``api_info`` dict mapping object ``id`` to a name set.

    """
    def __init__(self, module: ModuleType, ingore_keywords: List[str] = []) -> None:
        self._api_info: Dict[int, Set[str]] = {}
        self._top_module = module
        self._top_module_name: str = module.__name__
        self._ingore_keywords = set(ingore_keywords)
        self._build_api_info()
    
    def _build_api_info(self):

        visited_mem: Set[ModuleType] = set()

        def search_module(cur_module: ModuleType):

            visited_mem.add(cur_module)

            if not cur_module.__name__.startswith(self._top_module_name) \
                or not self._is_public_api(cur_module.__name__):
                return

            for module_name, module in getmembers(cur_module, ismodule):
                
                if module_name.startswith("_") or isbuiltin(module)  \
                    or self._ignore_name(module.__name__, self._ingore_keywords) \
                    or not module.__name__.startswith(self._top_module_name) \
                    or not self._is_public_api(cur_module.__name__):
                    continue

                prefix = module.__name__ + "."

                for predicate in [isfunction, isclass]:
                    for obj_name, obj in getmembers(module, predicate):
                        
                        if obj_name.startswith("_") or isbuiltin(obj) \
                            or self._ignore_name(obj.__module__, self._ingore_keywords) \
                            or not obj.__module__.startswith(self._top_module_name) \
                            or not self._is_public_api(obj.__module__):
                            continue

                        api_name, api_id = prefix + obj_name, id(obj)
                        if api_id not in self._api_info:
                            self._api_info[api_id] = set()
                        self._api_info[api_id].add(api_name)

                if module not in visited_mem:
                    search_module(module)

        search_module(self._top_module)

    def _ignore_name(self, name: str, ingore_keywords):
        for item in ingore_keywords:
            if item in name.split("."):
                return True
        return False

    def _is_public_api(self, input: str) -> bool:
        for s in input.split('.'):
            if s.startswith('_'):
                return False
        return True

    def get_api_info(self):
        return self._api_info


def get_generated_api_from_doc(path: str = None) -> Set[str]:
    """Return a list including name of APIs that exists in current document."""
    if path is None:
        script_path = os.path.dirname(os.path.realpath(__file__))
        build_path = os.path.join(script_path, "../build")
        assert os.path.isdir(build_path), "Please build the doc first"
        generated_api_path = os.path.join(build_path, "html", "reference", "api")
    else:
        generated_api_path = path

    generated_api = set()

    for (dirpath, dirnames, filenames) in os.walk(generated_api_path):
        for filename in filenames:
            generated_api.add(os.path.splitext(filename)[0])
    return generated_api

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Get unadded MegEngine Public APIs in doc.")
    parser.add_argument("-s", "--save", default=False, type=bool, help="save the output content to csv file")
    parser.add_argument("-v", "--verbose", default=True, type=bool, help="show detailss")

    args = parser.parse_args()

    # Step 1: Find generated APIs
    generated_api = get_generated_api_from_doc()

    # Step 2: See the whole world
    import megengine
    finder = PublicInterfaceFinder(megengine, 
        ingore_keywords = [
            "core", 
            "utils", 
            "traced_module"
        ],
    )
    api_info = finder.get_api_info()

    # Step 3: Find ungenerated APIs
    ungenerated_api = []

    for id, alias in api_info.items():
        generated = False
        for name in alias:
            if name in generated_api:
                generated = True
                break
        if not generated:
            ungenerated_api.append(list(alias))  # makes sorted output more human-friendly

    ungenerated_api.sort()

    # Now we have put the elephant into the fridge

    if args.save:
        import csv

        with open("/tmp/output.csv", "w") as file:
            wr = csv.writer(file, dialect='excel')
            wr.writerow(item for item in ungenerated_api)
        print("Saved in /tmp/output.csv")
    
    if args.verbose:
        for idx, item in enumerate(ungenerated_api):
            print(idx, item)