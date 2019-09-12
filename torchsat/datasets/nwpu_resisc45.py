from .folder import DatasetFolder
from .utils import default_loader

CLASSES_TO_IDX = {
    'airplane': 0,
    'airport': 1,
    'baseball_diamond': 2,
    'basketball_court': 3,
    'beach': 4,
    'bridge': 5,
    'chaparral': 6,
    'church': 7,
    'circular_farmland': 8,
    'cloud': 9,
    'commercial_area': 10,
    'dense_residential': 11,
    'desert': 12,
    'forest': 13,
    'freeway': 14,
    'golf_course': 15,
    'ground_track_field': 16,
    'harbor': 17,
    'industrial_area': 18,
    'intersection': 19,
    'island': 20,
    'lake': 21,
    'meadow': 22,
    'medium_residential': 23,
    'mobile_home_park': 24,
    'mountain': 25,
    'overpass': 26,
    'palace': 27,
    'parking_lot': 28,
    'railway': 29,
    'railway_station': 30,
    'rectangular_farmland': 31,
    'river': 32,
    'roundabout': 33,
    'runway': 34,
    'sea_ice': 35,
    'ship': 36,
    'snowberg': 37,
    'sparse_residential': 38,
    'stadium': 39,
    'storage_tank': 40,
    'tennis_court': 41,
    'terrace': 42,
    'thermal_power_station': 43,
    'wetland': 44,
}



class NWPU_RESISC45(DatasetFolder):

    url = 'https://sov8mq.dm.files.1drv.com/y4m_Fo6ujI52LiWHDzaRZVtkMIZxF7aqjX2q7KdVr329zVEurIO-wUjnqOAKHvHUAaoqCI0cjYlrlM7WCKVOLfjmUZz6KvN4FmV93qsaNIB9C8VN2AHp3JXOK-l1Dvqst8HzsSeOs-_5DOYMYspalpc1rt_TNAFtUQPsKylMWcdUMQ_n6SHRGRFPwJmSoJUOrOk2oXe9D7CPEq5cq9S9LI8hA/NWPU-RESISC45.rar?download&psid=1'

    def __init__(self, root, download=False, **kwargs):
        if download:
            download()
        
        extensions = ['.jpg', '.jpeg']
        classes = list(CLASSES_TO_IDX.keys())

        super(NWPU_RESISC45, self).__init__(root, default_loader, extensions,
        classes=classes, class_to_idx=CLASSES_TO_IDX , **kwargs)
        

    def download():
        pass
