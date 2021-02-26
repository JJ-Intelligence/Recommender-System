from progress.bar import Bar


class MasterBar(Bar):
    @property
    def remaining_minutes(self):
        return self.eta // 60

    @property
    def remaining_seconds(self):
        return self.eta % 60

    @property
    def elapsed_minutes(self):
        return self.elapsed // 60

    @property
    def elapsed_seconds(self):
        return self.elapsed % 60


class EpochBar(MasterBar):
    mse = 0
    message = 'Training'
    fill = '#'
    suffix = '%(index)d / %(max)d - elapsed %(elapsed_minutes)02d:%(elapsed_seconds)02d - eta %(' \
             'remaining_minutes)02d:%(remaining_seconds)02d - Eval mse %(mse)f'


class PercentageBar(MasterBar):
    message = 'Processing'
    fill = '#'
    suffix = '%(percent).1f%% - elapsed %(elapsed_minutes)02d:%(elapsed_seconds)02d - eta %(remaining_minutes)02d:%(' \
             'remaining_seconds)02d'