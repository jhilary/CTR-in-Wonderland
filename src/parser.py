from ciw.feature_types import PlainFeature, CategoricalFeature, Label, ID


class Record(object):
    def __init__(self, label, factors, record_id):
        self.label = label
        self.factors = factors
        self.record_id = record_id


class RecordsGenerator(object):
    def __init__(self, stream, records_filter=None):
        self.stream = stream
        self.counter = 0
        self.counter_filtered = 0
        self.records_filter = records_filter
        self.types = self.create_types(stream.readline())

    @staticmethod
    def create_types(headers):
        types = []
        for header in headers.strip().split(","):
            if header.startswith("CA") or header.startswith("AT"): #DAFUQ
                types.append((header, CategoricalFeature))
            elif header.startswith("NUM"):
                types.append((header, PlainFeature))
            elif header == "CLICK":
                types.append((header, Label))
            elif header == "ID":
                types.append((header, ID))
            else:
                raise ValueError("Bad header %s is given in headers" % header)
        return types

    @staticmethod
    def is_missing(value):
        return value == ""

    def __call__(self):
        for line in self.stream:
            self.counter += 1
            line_split = line.strip().split(",")
            features = {}
            label = None
            record_id = None
            for i in xrange(len(line_split)):
                element_value = line_split[i]
                if self.is_missing(element_value):
                    element_value = None

                namespace, feature_type = self.types[i]
                feature = feature_type(element_value)
                if isinstance(feature, Label):
                    label = feature
                elif isinstance(feature, ID):
                    record_id = feature
                else:
                    features[namespace] = feature
            record = Record(label,features,record_id)
            if self.records_filter is None or self.records_filter(record):
                self.counter_filtered += 1
                yield record