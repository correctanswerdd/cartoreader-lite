import os
import pytest
from cartoreader_lite import CartoStudy
from cartoreader_lite.low_level.study import CartoLLStudy
from cartoreader_lite.low_level.utils import read_point_data

study_dir = "openep-testingdata/Carto/Export_Study-1-11_25_2021-15-01-32"
study_name = "Study 1 11_25_2021 15-01-32.xml"

ecg_columns = "I(110)  II(111)  III(112)  aVL(171)  aVR(172)  aVF(173)  V1(21)  V2(22)  V3(23)  V4(24)  V5(25)  V6(26)  M1(1)  M2(2)"
ecg_values = " ".join(["1"] * 14)

def write_point(tmp_path, ecg_lines, ecg_attribs=""):
    with open(tmp_path / "1-Map_P1_ECG_Export.txt", "w") as ecg_f:
        ecg_f.write("\n".join(ecg_lines + [ecg_columns, ecg_values, ecg_values]) + "\n")
    with open(tmp_path / "1-Map_P1_Point_Export.xml", "w") as xml_f:
        xml_f.write(f'<Point><ECG FileName="1-Map_P1_ECG_Export.txt" {ecg_attribs}/></Point>')

def test_ecg_export_4_0(tmp_path):
    write_point(tmp_path, ["ECG_Export_4.0", "Raw ECG to MV (gain) = 0.003000",
                           "Unipolar Mapping Channel=M1 Bipolar Mapping Channel=M1-M2 Reference Channel=I"])
    metadata, data = read_point_data("1-Map", 1, str(tmp_path))
    ecg_metadata, ecg_data, channels = data["ecg"]
    assert len(ecg_metadata) == 3
    assert channels == ("M1", "M1-M2", "I")
    assert ecg_data.shape == (2, 14)

def test_ecg_export_4_1(tmp_path):
    write_point(tmp_path, ["ECG_Export_4.1", "Raw ECG to MV (gain) = 0.003000"],
                'UnipolarMappingChannel="M1" BipolarMappingChannel="M1-M2" ReferenceChannel="I"')
    metadata, data = read_point_data("1-Map", 1, str(tmp_path))
    ecg_metadata, ecg_data, channels = data["ecg"]
    assert len(ecg_metadata) == 2
    assert channels == ("M1", "M1-M2", "I")
    assert ecg_data.shape == (2, 14)

class TestStudyOptions():

    def test_map_names(self):
        all_maps = CartoLLStudy(study_dir, study_name).maps
        selected = all_maps[0].name
        assert len(CartoLLStudy(study_dir, study_name, map_names=selected).maps) == 1
        assert len(CartoLLStudy(study_dir, study_name, map_names=[selected]).maps) == 1
        assert len(CartoLLStudy(study_dir, study_name, map_names=lambda name: name == selected).maps) == 1
        assert len(CartoLLStudy(study_dir, study_name, map_names=[]).maps) == 0

    @pytest.mark.parametrize("executor", ["thread", None])
    def test_executor(self, executor):
        reference = CartoLLStudy(study_dir, study_name)
        study = CartoLLStudy(study_dir, study_name, executor=executor)
        assert [m.name for m in study.maps] == [m.name for m in reference.maps]
        assert [len(m.point_raw_data) for m in study.maps if hasattr(m, "point_raw_data")] == \
               [len(m.point_raw_data) for m in reference.maps if hasattr(m, "point_raw_data")]

    def test_point_cache(self, tmp_path):
        study = CartoLLStudy(study_dir, study_name, point_cache_dir=str(tmp_path))
        cached_maps = [m for m in study.maps if hasattr(m, "point_raw_data")]
        assert len(os.listdir(tmp_path)) == len(cached_maps)
        restored = CartoLLStudy(study_dir, study_name, point_cache_dir=str(tmp_path))
        assert [len(m.point_raw_data) for m in restored.maps if hasattr(m, "point_raw_data")] == \
               [len(m.point_raw_data) for m in cached_maps]

    def test_parse_ablation_and_columns(self):
        study = CartoStudy(study_dir, study_name, parse_ablation=False, carto_map_kwargs={"discard_invalid_points": False})
        assert study.ablation_data is None
        points = [m.points for m in study.maps if len(m.points) > 0]
        assert len(points) > 0
        for column in ["ecg_interest_channels", "surface_ecg", "egm", "proj_tri_id"]:
            assert column in points[0]
