class FeatureInfo():
    def __init__(self, weight, g_square):
        self.weight = weight
        self.g_square = g_square

class Label(object):
    def __init__(self, value):
        self.name = "label"
        self.value = int(value)
        if self.value not in {0, 1}:
            raise ValueError("Bad label %s provided" % value)
        self.info = None

class ID(object):
    def __init__(self, value):
        self.name = "ID"
        self.value = int(value)
        self.info = None


class CategoricalFeature(object):
    def __init__(self, value):
        self.info = None
        if value is None:
            self.name = "None"
            self.value = 1
        else:
            self.name = str(value)
            self.value = 1


class PlainFeature(object):
    def __init__(self, value):
        self.name = "0"
        self.info = None

        if value is None:
            self.value = None
        else:
            self.value = float(value)

    def __str__(self):
        return "%s:%s" % (self.name, self.value)