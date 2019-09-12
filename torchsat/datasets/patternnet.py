from .folder import DatasetFolder
from .utils import default_loader

CLASSES_TO_IDX = {
    'airplane': 0,
    'baseball_field': 1,
    'basketball_court': 2,
    'beach': 3,
    'bridge': 4,
    'cemetery': 5,
    'chaparral': 6,
    'christmas_tree_farm': 7,
    'closed_road': 8,
    'coastal_mansion': 9,
    'crosswalk': 10,
    'dense_residential': 11,
    'ferry_terminal': 12,
    'football_field': 13,
    'forest': 14,
    'freeway': 15,
    'golf_course': 16,
    'harbor': 17,
    'intersection': 18,
    'mobile_home_park': 19,
    'nursing_home': 20,
    'oil_gas_field': 21,
    'oil_well': 22,
    'overpass': 23,
    'parking_lot': 24,
    'parking_space': 25,
    'railway': 26,
    'river': 27,
    'runway': 28,
    'runway_marking': 29,
    'shipping_yard': 30,
    'solar_panel': 31,
    'sparse_residential': 32,
    'storage_tank': 33,
    'swimming_pool': 34,
    'tennis_court': 35,
    'transformer_station': 36,
    'wastewater_treatment_plant': 37,
}

class PatternNet(DatasetFolder):
    url = 'https://doc-0k-9c-docs.googleusercontent.com/docs/securesc/s4mst7k8sdlkn5gslv2v17dousor99pe/5kjb9nqbn6uv3dnpsqu7n7vbc2sjkm9n/1553925600000/13306064760021495251/10775530989497868365/127lxXYqzO6Bd0yZhvEbgIfz95HaEnr9K?e=download'
    
    def __init__(self, root, download=False, **kwargs):
        if download:
            download()
        
        extensions = ['.jpg']
        classes = list(CLASSES_TO_IDX.keys())

        super(PatternNet, self).__init__(root, default_loader, extensions,
        classes=classes ,class_to_idx=CLASSES_TO_IDX, **kwargs)
        

    def download():
        pass