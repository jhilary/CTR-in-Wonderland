
class LabelType(object):
    def __init__(self, name):
        self.name = name
    def __call__(self, value):
        return value


class CategoricalType(object):
    def __init__(self, name):
        self.name = name
    def __call__(self, value):
        if value is None:
            return {"None": 1}
        return {str(value): 1}


class PlainType(object):
    def __init__(self, name):
        self.name = name

    def __call__(self, value):
        if value is None:
            return {"0", None}
        return {"0", float(value)}


class RecordsGenerator(object):
    def __init__(self, stream):
        self.stream = stream
        self.types = self.create_types(stream.readline())

    @staticmethod
    def create_types(header):
        types = []
        for header in header.split(","):
            if header.startswith("CA") or header.startswith("AT"): #DAFUQ
                types.append(CategoricalType(header))
            elif header.startswith("NUM"):
                types.append(PlainType(header))
            elif header == "CLICK":
                types.append(LabelType(header))
            else:
                raise ValueError("Bad header %s is given in headers" % header)
        return types

    @staticmethod
    def is_missing(value):
        return value == ""

    def __call__(self):
        for line in self.stream:
            line_split = line.strip().split(",")
            result_data = {}
            label = None
            for i in xrange(len(line_split)):
                element_value = line_split[i]
                if self.is_missing(element_value):
                    element_value = None
                element_type = self.types[i]
                element = element_type(element_value)

                if isinstance(element_type, LabelType):
                    label = element
                else:
                    result_data[element_type.name] = element
            if label is None:
                print "Warning: missing label for line"
            yield result_data, label