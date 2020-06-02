import numpy as np


class Gait:
    def __init__(self, n_segment, offsets, durations, name):
        self.name = name

        self.stance = False

        self.n_iteration = n_segment

        self.offsets = offsets
        self.durations = durations
        self.offsetsFloat = np.array(offsets) / n_segment
        self.durationsFloat = np.array(durations) / n_segment

        # self.stance = durations[0]
        # self.swing = n_segment - durations[0]

        self.iteration = 0
        self.phase = 0.0  # range(0,1)

        self.stance = durations[0]

    def reset(self):
        self.phase = 0
        self.iteration = 0

    def get_contact_state(self):
        if self.stance:
            return [0.5, 0.5, 0.5, 0.5]

        progress = self.phase - self.offsetsFloat

        for i in range(4):
            if progress[i] < 0:
                progress[i] += 1
            if progress[i] > self.durationsFloat[i]:
                progress[i] = 0.0
            else:
                progress[i] = progress[i] / self.durationsFloat[i]

        return progress

    def get_swing_state(self):
        swing_offset = self.offsetsFloat + self.durationsFloat
        swing_offset = swing_offset - (swing_offset > 1)  # minus one for item > 1
        swing_duration = 1 - self.durationsFloat

        progress = self.phase - swing_offset
        progress = progress + (progress < 0)  # plus one for item < 0

        for i in range(4):
            if progress[i] > swing_duration[i]:
                progress[i] = 0.0
            else:
                progress[i] = progress[i] / swing_duration[i]

        return progress

    def step(self):
        self.iteration += 1
        if self.iteration == self.n_iteration:
            self.iteration = 0
        self.phase = self.iteration / self.n_iteration

    def get_current_iteration(self):
        return self.iteration

    def get_current_phase(self):
        return self.phase

