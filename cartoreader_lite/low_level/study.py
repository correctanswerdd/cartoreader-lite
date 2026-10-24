from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor, Future
import xml.etree.ElementTree as ET 
from xml.etree.ElementTree import Element
import os
import pandas as pd

from cartoreader_lite.low_level.read_mesh import read_mesh_file
from cartoreader_lite.low_level.visitags import read_visitag_dir
from .utils import camel_to_snake_case, read_point_data, xml_elem_to_dict, xml_to_dataframe
import numpy as np
from itertools import repeat
import tempfile
import zipfile
import pyvista as pv
from typing import Callable, Dict, Iterable, Union
import pickle

_parallelize_pool = ProcessPoolExecutor

class _SerialExecutor(Executor):
    """Executor running every task immediately in the calling thread"""

    def submit(self, fn, *args, **kwargs) -> Future:
        future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as ex:
            future.set_exception(ex)
        return future

def _make_pool(executor : str = "process") -> Executor:
    """Creates the executor used to parallelize the import.

    Parameters
    ----------
    executor : str, optional
        One of "process" (:class:`ProcessPoolExecutor`), "thread" (:class:`ThreadPoolExecutor`) or None (serial), by default "process"
    """
    if executor == "process":
        return _parallelize_pool()
    elif executor == "thread":
        return ThreadPoolExecutor()
    elif executor is None:
        return _SerialExecutor()
    else:
        raise ValueError(f"Unknown executor {executor}, expected 'process', 'thread' or None")

def _map_selected(map_name : str, map_names : Union[Iterable[str], Callable[[str], bool]]) -> bool:
    if map_names is None:
        return True
    elif callable(map_names):
        return map_names(map_name)
    else:
        return map_name in map_names

class CartoLLMap:

    """Low level CARTO3 map container for all point associated data.

    Parameters
    ----------
    xml_h : Element
        XML element holding the map data
    path_prefix : str
        Prefix of the path to load from
    point_cache_dir : str, optional
        Directory used to cache the raw point data of the map. If a cache file of this map exists, it is loaded
        instead of reading all point files, otherwise it is created after reading.
        Only point it to caches created by this library, since the files are loaded with :mod:`pickle`.
        By default None (no caching)
    executor : str, optional
        Executor used to read the points: "process", "thread" or None (serial). By default "process"
    """

    def import_raw_points(self, path_prefix : str, point_cache_dir : str = None, executor : str = "process"):
        """Imports all points and its detailed data of the current map

        Parameters
        ----------
        path_prefix : str
            Prefix of the path to load from
        point_cache_dir : str, optional
            Directory to cache the raw point data in, by default None (no caching)
        executor : str, optional
            Executor used to read the points: "process", "thread" or None (serial). By default "process"
        """

        cache_fname = None
        if point_cache_dir is not None:
            cache_fname = os.path.join(point_cache_dir, f"{self.name}_raw_points.pkl")
            if os.path.isfile(cache_fname):
                with open(cache_fname, "rb") as cache_f:
                    self.point_raw_data = pickle.load(cache_f)
                return

        #read_f = lambda point_id: read_point_data(self.name, point_id, path_prefix)
        with _make_pool(executor) as pool:
            self.point_raw_data = list(pool.map(read_point_data, repeat(self.name),
                                                        self.points_main_data["Id"], repeat(path_prefix),
                                                        chunksize=5))

        if cache_fname is not None:
            os.makedirs(point_cache_dir, exist_ok=True)
            with open(cache_fname, "wb") as cache_f:
                pickle.dump(self.point_raw_data, cache_f, protocol=pickle.HIGHEST_PROTOCOL)

    def __init__(self, xml_h : Element, path_prefix : str, point_cache_dir : str = None, executor : str = "process") -> None:

        for k, v in xml_h.items():
            setattr(self, camel_to_snake_case(k), v)

        for elem in xml_h:
            if elem.tag == "CartoPoints":
                self.points_main_data = xml_to_dataframe(elem)
                if "Position3D" in self.points_main_data:
                    self.points_main_data["Position3D"] = self.points_main_data["Position3D"].apply(lambda x: np.fromstring(x, sep=" "))
                if "CathOrientation" in self.points_main_data:
                    self.points_main_data["CathOrientation"] = self.points_main_data["CathOrientation"].apply(lambda x: np.fromstring(x, sep=" "))

                self.virtual_points = [[xml_elem_to_dict(sub_elem) for sub_elem in point if sub_elem.tag == "VirtualPoint"] for point in elem]
                self.point_tags = [[{**sub_elem.attrib, **{"value": int(sub_elem.text)}} for sub_elem in point if sub_elem.tag == "Tags"] for point in elem]
            elif elem.tag == "TagsTable":
                self.tags = xml_to_dataframe(elem)
            elif elem.tag == "RefAnnotationConfig":
                self.ref_annotation_config = xml_elem_to_dict(elem)
            elif elem.tag == "ColoringRangeTable":
                self.coloring_range_table = xml_to_dataframe(elem)

        #Import mesh
        if "FileNames" in xml_h.keys():
            fname = os.path.join(path_prefix, self.file_names)
            if os.path.isfile(fname):
                self.mesh, self.mesh_metadata = read_mesh_file(fname)
            else:
                print(f"Warning: File {fname} referenced for map {self.name}, but could not be found")
                raise FileNotFoundError(fname)
                #self.mesh = pv.UnstructuredGrid()
                #self.mesh_metadata = {}

        if "Id" in self.points_main_data: 
            self.import_raw_points(path_prefix, point_cache_dir, executor)

class CartoAuxMesh:
    """Class that holds auxiliary meshes of the CARTO system, e.g. generated by `CartoSeg`_.

    Parameters
    ----------
    xml_h : Element
        XML entry point from where to read the auxiliary meshes
    path_prefix : str
        Path prefix pointing to the directory to read from
    load : bool, optional
        If true, the meshes will be read and buffered immediately. 
        If false, only the names will be loaded and :meth:`load_mesh` will need to be called later.
        By default True
    """

    mesh_path : str #: Full path to the mesh file
    name : str #: Name of the mesh
    mesh_data : pv.UnstructuredGrid #: The loaded mesh
    metadata : Dict[str, str] #: Metadata associated with the mesh, given by the XML tags
    affine : np.ndarray #: 4x4 affine transformation matrix given by CARTO


    def __init__(self, xml_h : Element, path_prefix : str, load=True) -> None:
        for k, v in xml_h.items():
            setattr(self, camel_to_snake_case(k), v)

        self.mesh_path = os.path.join(path_prefix, xml_h.attrib["FileName"])
        self.name = os.path.splitext(self.file_name)[0]

        if load:
            self.load_mesh()

    def load_mesh(self):
        """Loads the mesh with the given name into the memory
        """
        self.mesh_data, self.metadata = read_mesh_file(self.mesh_path)
        if "Matrix" in self.metadata:
            self.affine = np.fromstring(self.metadata["Matrix"], sep=" ").reshape([4, 4])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, path={self.mesh_path}, mesh={self.mesh_data})"

class CartoLLStudy:
    """Low level CARTO study class that reads all information found in the CARTO3 study and saves it.

    Parameters
    ----------
    arg1 : str
        A path to either

            * A directory containing the study
            * A zip file with the study inside
    arg2 : str, optional
        The name of the study to load, contained inside the directory or zip file.
        Will default to either the zip name or bottom most directory name.
    map_names : Union[Iterable[str], Callable[[str], bool]], optional
        Only maps whose name is contained in this list (or for which this function returns True) will be imported.
        By default None (all maps)
    point_cache_dir : str, optional
        Directory used to cache the raw point data of each map, see :class:`CartoLLMap`. By default None (no caching)
    executor : str, optional
        Executor used to parallelize the import: "process", "thread" or None (serial). By default "process"
    """

    aux_mesh_reg_mat : np.ndarray = None

    def _parse_meshes(self, xml_h : Element, path_prefix : str):
        """Parses and loads the axuiliary meshes given in the study

        Parameters
        ----------
        xml_h : Element
            The XML element containing the mesh metadata
        path_prefix : str
            Path prefix pointing to the directory to read from
        """
        self.aux_meshes = []
        
        with _make_pool(self.executor) as pool:
            for elem in xml_h:
                if elem.tag == "RegistrationMatrix":
                    self.aux_mesh_reg_mat = np.fromstring(elem.text, sep=" ").reshape([4, 4]) #Affine matrix
                elif elem.tag == "Mesh":
                    self.aux_meshes.append(pool.submit(CartoAuxMesh, elem, path_prefix)) #CartoMesh(elem, path_prefix))
                elif elem.tag == "RegistrationData":
                    self.aux_mesh_reg_data = xml_elem_to_dict(elem)

            self.aux_meshes = [m.result() for m in self.aux_meshes]

    def _parse_maps(self, maps : Element, path_prefix : str):
        """Parses and loads the maps given in the study

        Parameters
        ----------
        maps : Element
            The XML element containing the map data
        path_prefix : str
            Path prefix pointing to the directory to read from
        """
        self.maps = []
        with _make_pool(self.executor) as pool:
            for elem in maps:
                if elem.tag == "Map":
                    if _map_selected(elem.attrib.get("Name"), self.map_names):
                        self.maps.append(pool.submit(CartoLLMap, elem, path_prefix, self.point_cache_dir, self.executor)) #self.maps.append(CartoLLMap(elem, path_prefix))
                elif elem.tag == "TagsTable":
                    self.tags_table = xml_to_dataframe(elem)
                elif elem.tag == "ColoringTable":
                    self.coloring_table = xml_to_dataframe(elem)

            maps = []
            for res in self.maps:
                try:
                    maps.append(res.result())
                except Exception as ex:
                    print(f"Importing a map failed. Original error: {type(ex)}, {ex}")
                    
            self.maps = maps

    def _read_xml(self, xml_h : ET, path_prefix : str):
        """Read the XML data of the study and parses all the data in it

        Parameters
        ----------
        xml_h : ET
            The XML root element of the study
        path_prefix : str
            Path prefix pointing to the directory to read from
        """
        root = xml_h.getroot()
        self.name = root.attrib["name"]
        futures = []
        with ThreadPoolExecutor() as pool:
            for elem in root:
                if elem.tag == "Maps":
                    futures.append(pool.submit(self._parse_maps, elem, path_prefix)) #self._parse_maps(elem, path_prefix)
                elif elem.tag == "Meshes":
                    futures.append(pool.submit(self._parse_meshes, elem, path_prefix))

            [res.result() for res in futures]

    def _from_zip(self, zip_fname : str, study_name : str = None):
        """Loads the study from a zipped file by extracting it first and then calling :meth:`._from_dir`

        Parameters
        ----------
        zip_fname : str
            The name of the zip file
        study_name : str, optional
            The name of the XML study inside the zip file. Can also contain sub-folder paths.
            Will default to the name of the zip file.
        """
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            with zipfile.ZipFile(zip_fname, "r") as zip_f:
                zip_f.extractall(tmp_dir_name)

            self._from_dir(tmp_dir_name, study_name)

    def _from_dir(self, dir_name : str, study_name : str = None):
        """Loads the study from a directory file

        Parameters
        ----------
        dir_name : str
            Name of the directory to read from
        study_name : str, optional
            The name of the XML study inside the zip file.
            Will default to the name of the bottom-most directory.
        """
        if study_name is None:
            study_name = os.path.basename(os.path.normpath(dir_name))

        if not study_name.endswith(".xml"):
             study_name += ".xml"

        #Move any folder directives from the study name to the dir name
        dir_name = os.path.join(dir_name, os.path.split(study_name)[0])
        study_name = os.path.split(study_name)[1]

        self.aux_meshes = []        
        full_fname = os.path.join(dir_name, study_name)
        # Pass the path of the xml document 
        study_xml = ET.parse(full_fname) 
        self._read_xml(study_xml, dir_name)
        #study_root = study_xml.getroot()
        self.visitag_data = read_visitag_dir(os.path.join(dir_name, "VisiTagExport"))

    def __init__(self, arg1 : str, arg2 : str = None, map_names : Union[Iterable[str], Callable[[str], bool]] = None,
                 point_cache_dir : str = None, executor : str = "process") -> None:
        assert issubclass(type(arg1), str), "Given arguments not (yet) supported"
        if isinstance(map_names, str):
            map_names = [map_names]
        self.map_names = map_names
        self.point_cache_dir = point_cache_dir
        self.executor = executor
        if os.path.isdir(arg1):
            self._from_dir(arg1, arg2)
        elif os.path.isfile(arg1) and arg1.endswith(".zip"): #Possible second argument: study name
            self._from_zip(arg1, arg2)
        else:
            assert False, "Given arguments not (yet) supported, or the study file/folder was not found."
