import time

import cv2
import numpy as np
from PIL import ImageGrab
from selenium import webdriver
from selenium.webdriver.common.keys import Keys


class Game:
    # scaled down to 20 pixels + 4 bit score + jump state
    # After score 1700 there is no change in speed
    observation_space_n = 2**25
    # Do nothing, jump
    action_space_n = 2

    # First state bit 1
    jump_num = 2**24

    def __init__(self, score_offset: int = 0,
                 no_acceleration: bool = False) -> None:

        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument("--disable-infobars")
        chrome_options.add_argument("--mute-audio")
        self._driver = webdriver.Chrome(options=chrome_options)

        self._driver.set_window_position(-10, 0)
        self._driver.set_window_size(670, 240)

        self._driver.get('chrome://dino/')
        time.sleep(1)

        self.score_offset = score_offset
        if self.score_offset != 0:
            set_speed = self._score_offset_to_speed(self.score_offset)
            self._driver.execute_script(f'Runner.config.SPEED={set_speed}')

        self.no_acceleration = no_acceleration
        if self.no_acceleration:
            self._driver.execute_script('Runner.config.ACCELERATION=0')

    def start(self):
        self.jump()

    @staticmethod
    def _score_offset_to_speed(score_offset: int) -> float:
        """ Values were acquired experimentally.
        """
        if not 0 <= score_offset <= 1000:
            raise ValueError(
                f"score_offset must be between 0 and 1000: {score_offset}")
        if score_offset % 100:
            raise ValueError(
                f"score_offset is not round number: {score_offset}")

        if score_offset == 0:
            return 6.0
        elif score_offset == 100:
            return 6.66
        elif score_offset == 200:
            return 7.2
        elif score_offset == 300:
            return 7.75
        elif score_offset == 400:
            return 8.2
        elif score_offset == 500:
            return 8.75
        elif score_offset == 600:
            return 9.17
        elif score_offset == 700:
            return 9.6
        elif score_offset == 800:
            return 10.0
        elif score_offset == 900:
            return 10.4
        elif score_offset == 1000:
            return 10.8

    @staticmethod
    def _scale_down(state: np.array, new_len: int = 20) -> np.array:
        # assert len(state) == 512
        step_size = len(state) // new_len
        res = np.zeros(new_len)

        for i in range(new_len):
            a = i * step_size
            b = a + step_size
            if state[a:b].any():
                res[i] = 1

        return res

    @staticmethod
    def _detect_obstacles(gray: np.array) -> np.array:
        _, ix = gray.shape
        res = [0] * ix
        res = np.array(res, dtype=np.uint8)
        for x in range(ix):
            column = gray[:, x]
            if not column.all():
                res[x] = 255

        return res

    @staticmethod
    def _detect_obstacles_scale_down(gray: np.array, new_len: int = 20) -> np.array:
        _, ix = gray.shape
        step_size = ix // new_len
        res = np.zeros(new_len, dtype=np.uint8)
        for i in range(new_len):
            a = i * step_size
            b = a + step_size
            if not gray[:, a:b].all():
                res[i] = 1

        return res

    def get_state(self) -> int:
        bbox_x, bbox_y = 112, 111
        bbox_width, bbox_height = 492, 108
        image = np.array(ImageGrab.grab(
            bbox=(bbox_x, bbox_y, bbox_x+bbox_width, bbox_y+bbox_height)))

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]

        if np.median(gray[0, :]) == 0:
            gray = np.invert(gray)

        state = self._detect_obstacles_scale_down(gray).tolist()
        # Convert binary list to in int
        state_num = int(''.join(str(int(x)) for x in state), 2)

        # Append additional info: score
        if not self.no_acceleration:
            current_score_state = min(self.get_score() // 100, 15)
            state_num = state_num << 4 | current_score_state

        # Append additional info: is jumping
        state_num = state_num << 1 | int(self.is_jumping())

        return state_num

    def take_action(self, action: int) -> tuple:
        if action == 1:
            self.jump()

        state = self.get_state()
        game_over = self.is_game_over()

        if game_over:
            reward = -1000
        # Is jumping
        elif (state % 2 == 1):
            # Reward for jumping over stuff
            if state >= self.jump_num:
                reward = 5
            # Penalty for just jumping
            else:
                reward = -1
        # Reward for living/staying on ground
        else:
            reward = 1

        return state, reward, game_over

    def jump(self) -> None:
        self._driver.find_element_by_tag_name('body').send_keys(Keys.SPACE)

    def is_jumping(self) -> bool:
        return self._driver.execute_script("return Runner.instance_.tRex.jumping")

    def is_game_over(self) -> bool:
        return self._driver.execute_script('return Runner.instance_.crashed')

    def get_score(self) -> None:
        score_array = self._driver.execute_script(
            "return Runner.instance_.distanceMeter.digits")
        score = ''.join(score_array)
        return int(score) + self.score_offset

    def reset(self) -> list:
        self._driver.execute_script('Runner.instance_.restart()')
        time.sleep(.5)
        return self.get_state()

    def end(self) -> None:
        self._driver.quit()
