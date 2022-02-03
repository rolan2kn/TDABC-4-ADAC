

class SimplicialComplexType:
    RIPS, ALPHA, WITNESS, TANGENTIAL, COVER, COSINE, LSH, \
    ENSEMBLE, MHM = range(9)

    def __init__(self, type = None, secondary_type = None, max_dim=3, max_value = None):
        self.complex_type = type if type is not None else SimplicialComplexType.RIPS
        self.max_dim = max_dim
        self.max_value = max_value
        if type == SimplicialComplexType.ENSEMBLE:
            if secondary_type is None:
                secondary_type = SimplicialComplexType.RIPS
            elif isinstance(secondary_type, SimplicialComplexType):
                secondary_type = secondary_type.get_type()

        self.secondary_type = secondary_type

    def get_maximal_dimension(self):
        return self.max_dim

    def get_maximum_value(self):
        return self.max_value

    def to_str(self, ctype=None):
        if ctype is None:
            ctype = self.complex_type

        if ctype == SimplicialComplexType.RIPS:
            return "RIPS"
        elif ctype == SimplicialComplexType.ALPHA:
            return "ALPHA"
        elif ctype == SimplicialComplexType.WITNESS:
            return "WITNESS"
        elif ctype == SimplicialComplexType.TANGENTIAL:
            return "TANGENTIAL"
        elif ctype == SimplicialComplexType.COVER:
            return "COVER"
        elif ctype == SimplicialComplexType.COSINE:
            return "COSINE"
        elif ctype == SimplicialComplexType.LSH:
            return "LSH"
        elif ctype == SimplicialComplexType.MHM:
            return "MHM"

        return "ENSEMBLE"

    def get_type(self):
        return self.complex_type

    def get_name(self):
        return self.to_str(self.complex_type)

    def get_secondary_type(self):
        return SimplicialComplexType(type=self.secondary_type)

    def is_current_sc_type(self, ctype):
        return self.complex_type == ctype

    def is_secondary_sc_type(self, ctype):
        return self.complex_type == SimplicialComplexType.ENSEMBLE and self.secondary_type == ctype

    def is_RIPS(self):
        return self.complex_type == SimplicialComplexType.RIPS or self.is_secondary_sc_type(SimplicialComplexType.RIPS)

    def is_ALPHA(self):
        return self.complex_type == SimplicialComplexType.ALPHA or self.is_secondary_sc_type(SimplicialComplexType.ALPHA)

    def is_WITNESS(self):
        return self.complex_type == SimplicialComplexType.WITNESS or self.is_secondary_sc_type(SimplicialComplexType.WITNESS)

    def is_TANGENTIAL(self):
        return self.complex_type == SimplicialComplexType.TANGENTIAL or self.is_secondary_sc_type(SimplicialComplexType.TANGENTIAL)

    def is_COVER(self):
        return self.complex_type == SimplicialComplexType.COVER or self.is_secondary_sc_type(SimplicialComplexType.COVER)

    def is_COSINE(self):
        return self.complex_type == SimplicialComplexType.COSINE or self.is_secondary_sc_type(SimplicialComplexType.COSINE)

    def is_LSH(self):
        return self.complex_type == SimplicialComplexType.LSH or self.is_secondary_sc_type(SimplicialComplexType.LSH)

    def is_ENSEMBLE(self):
        return self.complex_type == SimplicialComplexType.ENSEMBLE

    def is_MHM(self):
        return self.complex_type == SimplicialComplexType.MHM
