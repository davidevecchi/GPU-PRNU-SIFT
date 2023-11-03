
class DeviceData:
    device_data = {
        # original order: 02, 20, 29, 34, 05, 14, 06, 19, 25, 10, 18, 15, 12
        'D02': {'basescaling': 1.3331740000000000, 'crop_array': [345, 1491, 206, 2242], 'model': 'Apple_iPhone4s'},
        'D05': {'basescaling': 1.4548940000000000, 'crop_array': [269, 1414, 103, 2140], 'model': 'Apple_iPhone5c'},
        'D06': {'basescaling': 1.4168480000000000, 'crop_array': [291, 1437, 134, 2170], 'model': 'Apple_iPhone6'},
        'D10': {'basescaling': 1.3331230000000000, 'crop_array': [345, 1491, 206, 2242], 'model': 'Apple_iPhone4s'},
        'D14': {'basescaling': 1.4548600000000000, 'crop_array': [269, 1414, 103, 2140], 'model': 'Apple_iPhone5c'},
        'D15': {'basescaling': 1.4165250000000000, 'crop_array': [291, 1437, 134, 2170], 'model': 'Apple_iPhone6'},
        'D18': {'basescaling': 1.4548260000000000, 'crop_array': [269, 1414, 104, 2140], 'model': 'Apple_iPhone5c'},
        'D19': {'basescaling': 1.4172390000000000, 'crop_array': [291, 1436, 133, 2170], 'model': 'Apple_iPhone6Plus'},
        'D20': {'basescaling': 1.2270420000000004, 'crop_array': [216, 1362, 38, 2074], 'model': 'Apple_iPadMini'},
        'D25': {'basescaling': 1.9331158333333300, 'crop_array': [327, 1473, 182, 2218], 'model': 'OnePlus_A3000'},
        'D29': {'basescaling': 1.4548940000000000, 'crop_array': [269, 1414, 103, 2140], 'model': 'Apple_iPhone5'},
        'D34': {'basescaling': 1.4548770000000000, 'crop_array': [269, 1414, 103, 2140], 'model': 'Apple_iPhone5'},
        # 'D12': {'basescaling': 2.6385200000000000, 'crop_array': [460, 1988, 275, 2989], 'model': 'Sony_XperiaZ1Compact'},
    }
    
    def __init__(self, fingerprint_path, device_h0=None):
        device = fingerprint_path if device_h0 is None else device_h0
        self.id = device[-7:-4]
        self.basescaling = self.device_data[self.id]['basescaling']
        self.crop_array = self.device_data[self.id]['crop_array']  # [off_x, size_x, off_y, size_y]
