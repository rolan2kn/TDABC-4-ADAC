

class PersistenceIntervalStage:
    BIRTH, MIDDLE, DEATH = range(3)

    def __init__(self, stage=None):
        if stage is None:
            stage = PersistenceIntervalStage.DEATH
        self.stage = stage

    def get_stage(self):
        return self.stage

    def to_str(self, pistage=None):
        if pistage is None:
            pistage = self.stage

        if pistage == PersistenceIntervalStage.BIRTH:
            return "BIRTH"
        elif pistage == PersistenceIntervalStage.MIDDLE:
            return "MIDDLE"

        return "DEATH"

    def first_letter(self, pistage=None):
        if pistage is None:
            pistage = self.stage

        name = self.to_str(pistage)

        if len(name) == 0:
            return ""

        return name[0]

    def stages(self):
        return [PersistenceIntervalStage.BIRTH, PersistenceIntervalStage.MIDDLE, PersistenceIntervalStage.DEATH]

    def is_Birth(self, choice):
        return choice == PersistenceIntervalStage.BIRTH

    def is_Middle(self, choice):
        return choice == PersistenceIntervalStage.MIDDLE

    def is_Death(self, choice):
        return choice == PersistenceIntervalStage.DEATH