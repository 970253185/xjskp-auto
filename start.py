import os
import time
import random
import json5
import cv2
import numpy as np
import pygetwindow as gw
import pyautogui
import pytesseract
import logging
from logging.handlers import RotatingFileHandler
from mss import mss
from PIL import Image
import sys
import ctypes

# 全局配置变量
config = None

# 全局调试模式标志
DEBUG_MODE = False

# 检查管理员权限
def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

# 如果不是管理员，请求提升权限
if not is_admin():
    # 重新以管理员权限运行脚本
    ctypes.windll.shell32.ShellExecuteW(
        None, "runas", sys.executable, " ".join(sys.argv), None, 1
    )
    sys.exit(0)

# 设置日志
def setup_logging(log_file_path):
    """设置日志记录，同时输出到控制台和文件"""
    # 创建日志目录（如果不存在）
    log_dir = os.path.dirname(log_file_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 格式化当前日期
    from datetime import datetime
    formatted_path = datetime.now().strftime(log_file_path)
    
    # 创建日志记录器
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if DEBUG_MODE else logging.INFO)
    
    # 清除现有的处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 创建文件处理器
    file_handler = RotatingFileHandler(
        formatted_path, 
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG if DEBUG_MODE else logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if DEBUG_MODE else logging.INFO)
    
    # 创建格式化器
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 设置格式化器
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器到记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return formatted_path

# 日志输出函数
def print_info(message):
    """打印信息日志"""
    logging.info(message)

def print_error(message):
    """打印错误日志"""
    logging.error(message)

def print_debug(message):
    """打印调试日志（只在调试模式下显示）"""
    if DEBUG_MODE:
        logging.debug(message)

def load_config(config_path):
    """加载并验证配置文件"""
    global config
    
    if not os.path.exists(config_path):
        print_error(f"配置文件不存在：{config_path}")
        exit(1)
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json5.load(f)
    except Exception as e:
        print_error(f"加载配置文件失败：{e}")
        exit(1)
    
    # 验证必要配置项
    required_keys = [
        "template_root", "tesseract_path", "skill_template_paths",
        "start_button_templates", "exit_button_templates", "back_button_templates",
        "pause_button_templates", "match_threshold", "match_method", "check_interval",
        "max_wait_seconds", "target_level", "loop_count", "priority_skill_patterns",
        "sleep_times", "retry_settings", "calibration", "image_processing",
        "pause_button", "exit_button", "back_button", "start_button", "game_title"
    ]
    
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        print_error(f"配置文件缺少必要项：{', '.join(missing_keys)}")
        exit(1)
    
    # 检查模板根目录是否存在
    if not os.path.exists(config["template_root"]):
        print_error(f"模板文件夹不存在：{config['template_root']}")
        print_error("请创建该文件夹并放入所需的模板图片")
        exit(1)
    
    return config

def get_full_template_path(filename):
    """获取模板文件的完整路径"""
    full_path = os.path.join(config["template_root"], filename)
    return os.path.abspath(full_path)

def check_all_templates():
    """检查所有配置的模板图片是否存在"""
    missing_templates = []
    
    # 检查所有类型的模板
    all_templates = (
        config["start_button_templates"] +
        config["skill_template_paths"] +
        config["exit_button_templates"] +
        config["back_button_templates"] +
        config["pause_button_templates"]
    )
    
    for filename in all_templates:
        full_path = get_full_template_path(filename)
        if not os.path.exists(full_path):
            missing_templates.append(full_path)
    
    return missing_templates

def check_tesseract(tesseract_path):
    """检查Tesseract是否可用"""
    try:
        # 设置Tesseract路径
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        # 测试Tesseract是否正常工作
        test_img = np.zeros((100, 100), np.uint8)
        cv2.putText(test_img, "Test", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        pytesseract.image_to_string(test_img)
        return True
    except Exception as e:
        print_error(f"Tesseract OCR检查失败：{e}")
        return False

def get_window_info(window_title):
    """获取游戏窗口信息，支持模糊匹配"""
    try:
        # 获取所有窗口
        all_windows = gw.getAllWindows()
        matching_windows = []
        
        # 模糊匹配窗口标题
        for window in all_windows:
            if window_title.lower() in window.title.lower():
                matching_windows.append(window)
                print_debug(f"找到匹配窗口: {window.title}")
        
        if not matching_windows:
            print_info(f"未找到标题包含 '{window_title}' 的窗口")
            # 列出所有窗口标题以便调试
            print_debug("当前所有窗口标题:")
            for window in all_windows:
                if window.title:  # 只显示有标题的窗口
                    print_debug(f"- '{window.title}'")
            return None
        
        # 优先选择活动窗口
        active_window = None
        for window in matching_windows:
            if window.isActive:
                active_window = window
                break
        
        # 如果没有活动窗口，选择第一个匹配窗口
        if active_window is None:
            active_window = matching_windows[0]
            print_info(f"找到 {len(matching_windows)} 个匹配窗口，使用第一个窗口: {active_window.title}")
        
        # 确保窗口可见
        if active_window.isMinimized:
            print_info("窗口已最小化，尝试恢复")
            active_window.restore()
            time.sleep(0.5)
        
        return {
            "window": active_window,
            "title": active_window.title,
            "left": active_window.left,
            "top": active_window.top,
            "width": active_window.width,
            "height": active_window.height,
            "is_active": active_window.isActive
        }
    except Exception as e:
        print_error(f"获取窗口信息失败：{e}")
        return None

def check_window_foreground(window_info):
    """检查窗口是否在前台，不在则尝试激活"""
    try:
        max_attempts = 3
        for attempt in range(max_attempts):
            if window_info and window_info["is_active"]:
                print_info("游戏窗口已在前台")
                return window_info
            
            print_info(f"游戏窗口不在前台，尝试激活... (尝试 {attempt+1}/{max_attempts})")
            
            # 尝试激活窗口
            try:
                window_info["window"].activate()
            except Exception as e:
                print_error(f"窗口激活失败: {e}")
                # 尝试通过点击窗口区域来激活
                center_x = window_info["left"] + window_info["width"] // 2
                center_y = window_info["top"] + window_info["height"] // 2
                pyautogui.click(center_x, center_y)
            
            # 等待窗口激活
            time.sleep(config["sleep_times"]["window_activation"])
            
            # 重新获取窗口信息检查激活状态
            updated_window = get_window_info(config["game_title"])
            if updated_window and updated_window["is_active"]:
                print_info("游戏窗口激活成功，已处于前台")
                return updated_window
            
            # 如果还有尝试机会，等待一段时间再试
            if attempt < max_attempts - 1:
                time.sleep(1)
        
        print_error("游戏窗口激活失败，无法将其置于前台")
        return None
    except Exception as e:
        print_error(f"检查/激活窗口失败：{e}")
        return None

def capture_screenshot(window_info):
    """捕获游戏窗口的截图"""
    try:
        with mss() as sct:
            monitor = {
                "top": window_info["top"],
                "left": window_info["left"],
                "width": window_info["width"],
                "height": window_info["height"]
            }
            sct_img = sct.grab(monitor)
            img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
            return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print_error(f"截图失败：{e}")
        return None

def preprocess_image(image):
    """预处理图像以提高识别率"""
    # 转换为灰度图
    if config["image_processing"]["use_gray"] and len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 调整对比度
    if config["image_processing"]["contrast"] != 1.0:
        image = cv2.convertScaleAbs(
            image, 
            alpha=config["image_processing"]["contrast"], 
            beta=0
        )
    
    # 应用阈值
    if config["image_processing"]["threshold"] >= 0:
        _, image = cv2.threshold(
            image, 
            config["image_processing"]["threshold"], 
            255, 
            cv2.THRESH_BINARY
        )
    
    return image

def multi_template_match(screenshot, template_filenames, threshold=None, region=None):
    """多模板匹配，返回最佳匹配结果"""
    if threshold is None:
        threshold = config["match_threshold"]
        
    best_val = -1
    best_pos = None
    best_size = None
    match_methods = {
        "TM_CCOEFF": cv2.TM_CCOEFF,
        "TM_CCOEFF_NORMED": cv2.TM_CCOEFF_NORMED,
        "TM_CCORR": cv2.TM_CCORR,
        "TM_CCORR_NORMED": cv2.TM_CCORR_NORMED,
        "TM_SQDIFF": cv2.TM_SQDIFF,
        "TM_SQDIFF_NORMED": cv2.TM_SQDIFF_NORMED
    }
    method = match_methods.get(config["match_method"], cv2.TM_CCOEFF_NORMED)
    
    # 如果指定了区域，则截取该区域
    if region:
        x, y, w, h = region
        screenshot = screenshot[y:y+h, x:x+w]
    
    for filename in template_filenames:
        full_path = get_full_template_path(filename)
        
        if not os.path.exists(full_path):
            print_debug(f"模板文件不存在：{full_path}")
            continue
            
        try:
            # 读取模板图片
            template = cv2.imread(full_path)
            if template is None:
                print_error(f"→ OpenCV读取失败（可能格式错误）")
                continue
            
            # 预处理模板和截图
            template = preprocess_image(template)
            processed_screenshot = preprocess_image(screenshot.copy())
            
            # 确保模板尺寸小于截图
            if (template.shape[0] > processed_screenshot.shape[0] or 
                template.shape[1] > processed_screenshot.shape[1]):
                print_debug(f"→ 模板尺寸大于截图，跳过")
                continue
            
            # 执行匹配
            result = cv2.matchTemplate(processed_screenshot, template, method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            # 根据匹配方法确定最佳值
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                current_val = 1 - min_val  # 转换为越高越好的评分
                current_loc = min_loc
            else:
                current_val = max_val
                current_loc = max_loc
                
            # 只在调试模式下显示匹配值
            print_debug(f"→ {filename} 匹配值：{current_val:.4f}（阈值：{threshold}）")
            
            # 更新最佳匹配
            if current_val > best_val and current_val >= threshold:
                best_val = current_val
                h, w = template.shape[:2]
                # 如果指定了区域，需要调整坐标
                if region:
                    rx, ry, rw, rh = region
                    best_pos = (rx + current_loc[0] + w // 2, ry + current_loc[1] + h // 2)
                else:
                    best_pos = (current_loc[0] + w // 2, current_loc[1] + h // 2)
                best_size = (w, h)
                print_debug(f"→ 更新最佳匹配（模板：{filename}，值：{best_val:.4f}，位置：{best_pos}）")
                
        except Exception as e:
            print_error(f"→ 处理模板 {filename} 时出错：{e}")
            continue
    
    # 最终匹配结果
    if best_val >= threshold and best_pos:
        print_debug(f"最终最佳匹配：值={best_val:.4f}，位置={best_pos}")
        return (True, best_pos, best_val, best_size)
    else:
        print_debug(f"未找到符合阈值的匹配（最佳值：{best_val:.4f} < 阈值：{threshold}）")
        return (False, None, best_val, None)

def click_position(window_info, x, y, x_offset=0, y_offset=0, description="未知位置"):
    """点击窗口中的指定位置"""
    try:
        # 添加随机延迟，模拟人类操作
        delay = random.uniform(
            config["sleep_times"]["before_click_delay_min"],
            config["sleep_times"]["before_click_delay_max"]
        )
        time.sleep(delay)
        
        # 计算绝对坐标
        abs_x = window_info["left"] + x + x_offset
        abs_y = window_info["top"] + y + y_offset
        
        # 移动鼠标并点击
        move_duration = random.uniform(
            config["sleep_times"]["move_duration_min"],
            config["sleep_times"]["move_duration_max"]
        )
        pyautogui.moveTo(abs_x, abs_y, duration=move_duration)
        pyautogui.click()
        
        print_info(f"点击 {description}：窗口内({x}, {y}) → 绝对坐标({abs_x}, {abs_y})")
        return True
    except Exception as e:
        print_error(f"点击 {description} 失败：{e}")
        return False

def verify_start_game_success(window_info):
    """验证开始游戏是否成功（检查开始按钮是否消失）"""
    max_checks = config["retry_settings"]["start_verification_max_checks"]
    check_interval = config["sleep_times"]["start_verification_interval"]
    
    for i in range(max_checks):
        screenshot = capture_screenshot(window_info)
        if screenshot is None:
            time.sleep(check_interval)
            continue
            
        # 检查开始按钮是否仍然存在
        match_result, _, _, _ = multi_template_match(
            screenshot, 
            config["start_button_templates"],
            threshold=0.6  # 提高阈值，减少误判
        )
        
        if not match_result:
            print_info("开始游戏成功，开始按钮已消失")
            return True
            
        print_info(f"等待开始游戏完成（{i+1}/{max_checks}）")
        time.sleep(check_interval)
    
    print_info("开始游戏验证失败，可能未成功进入游戏")
    return False

def click_start_game_with_retry():
    """带重试机制的点击开始游戏按钮"""
    max_attempts = config["retry_settings"]["start_game_max_attempts"]
    attempt = 0
    
    # 使用开始按钮特定的阈值，如果配置中有的话
    start_threshold = config.get("start_button_threshold", 0.6)  # 默认0.6
    
    while attempt < max_attempts:
        attempt += 1
        print_info(f"\n===== 第{attempt}/{max_attempts}次尝试：寻找并点击「开始游戏」 =====")
        
        # 获取窗口信息
        window_info = get_window_info(config["game_title"])
        if not window_info:
            print_error("无法获取窗口信息，将重试")
            time.sleep(config["sleep_times"]["retry_interval"])
            continue

        # 确保窗口在前台
        window_info = check_window_foreground(window_info)
        if not window_info:
            print_error("窗口激活失败，将重试")
            time.sleep(config["sleep_times"]["retry_interval"])
            continue

        # 截取当前屏幕
        screenshot = capture_screenshot(window_info)
        if screenshot is None:
            print_error("截图失败，将重试")
            time.sleep(config["sleep_times"]["retry_interval"])
            continue

        # 尝试匹配开始按钮，使用开始按钮特定的阈值
        print_debug(f"开始按钮模板列表：{config['start_button_templates']}")
        match_result, match_pos, match_val, match_size = multi_template_match(
            screenshot, 
            config["start_button_templates"],
            threshold=start_threshold  # 使用开始按钮特定的阈值
        )
        
        # 处理匹配结果
        if match_result and match_pos:
            print_info(f"找到开始按钮（匹配值：{match_val:.2f}）")
            click_success = click_position(
                window_info, 
                match_pos[0], match_pos[1], 
                config["calibration"]["start_x"], 
                config["calibration"]["start_y"], 
                "开始按钮（图片匹配）"
            )
            if click_success:
                time.sleep(config["sleep_times"]["after_start_game_click"])
                # 验证点击是否成功
                if verify_start_game_success(window_info):
                    return window_info
        else:
            print_info(f"未找到匹配的开始按钮（匹配阈值：{start_threshold}），尝试默认位置")
            # 点击默认位置
            x = int(window_info["width"] * config["start_button"]["default_x_ratio"])
            y = int(window_info["height"] * config["start_button"]["default_y_ratio"])
            click_success = click_position(
                window_info, x, y, 
                config["calibration"]["start_x"], 
                config["calibration"]["start_y"], 
                "开始按钮（默认位置）"
            )
            if click_success:
                time.sleep(config["sleep_times"]["after_start_game_click"])
                # 验证点击是否成功
                if verify_start_game_success(window_info):
                    return window_info
        
        # 准备下一次尝试
        if attempt < max_attempts:
            print_info(f"第{attempt}次尝试失败，{config['sleep_times']['retry_interval']}秒后重试...")
            time.sleep(config["sleep_times"]["retry_interval"])

    print_error(f"已尝试{max_attempts}次点击开始游戏，均未成功")
    return None

def detect_skill_selection_screen(window_info):
    """检测是否进入选择技能界面"""
    screenshot = capture_screenshot(window_info)
    if screenshot is None:
        return False
        
    # 正向匹配：检测技能界面特征
    match_result, _, match_val, _ = multi_template_match(
        screenshot, 
        config["skill_template_paths"]
    )
    
    # 反向验证：检查开始按钮是否消失
    start_match, _, _, _ = multi_template_match(
        screenshot, 
        config["start_button_templates"],
        threshold=0.6
    )
    
    # 只有技能界面特征存在且开始按钮不存在，才判定为技能界面
    is_skill_screen = match_result and not start_match
    
    # 只在调试模式下输出详细匹配信息
    print_debug(f"技能界面检测：{'存在' if is_skill_screen else '不存在'}（匹配值：{match_val:.2f}）")
    
    return is_skill_screen

def wait_for_skill_selection_screen(window_info):
    """等待进入选择技能界面"""
    print_info(f"等待进入「选择技能」界面（最多等待{config['max_wait_seconds']}秒）")
    start_time = time.time()
    
    # 只在调试模式下输出检查间隔信息
    print_debug(f"每{config['check_interval']}秒检查一次技能界面")
    
    while time.time() - start_time < config["max_wait_seconds"]:
        if detect_skill_selection_screen(window_info):
            print_info("已进入「选择技能」界面")
            return True
            
        # 检查是否超时
        elapsed = int(time.time() - start_time)
        remaining = int(config["max_wait_seconds"] - elapsed)
        if remaining <= 0:
            break
            
        time.sleep(config["check_interval"])
    
    print_error(f"等待「选择技能」界面超时（{config['max_wait_seconds']}秒）")
    return False

def select_priority_skill(window_info):
    """选择优先级最高的技能"""
    print_info("开始选择技能...")
    screenshot = capture_screenshot(window_info)
    if screenshot is None:
        return False
    
    # 尝试匹配优先级最高的技能
    for skill_pattern in config["priority_skill_patterns"]:
        print_debug(f"尝试查找优先级技能: {skill_pattern}")
        
        # 这里简化处理，实际应该根据技能名称进行OCR识别
        # 或者为每个优先级技能准备模板图片
        # 目前只是简单地点击屏幕中间偏下位置
        
        # 等待一段时间让技能界面稳定
        time.sleep(0.5)
        
        # 重新截取屏幕，确保是最新状态
        screenshot = capture_screenshot(window_info)
        if screenshot is None:
            continue
            
        # 这里可以添加实际的技能识别逻辑
        # 例如使用OCR识别技能名称，或者使用模板匹配特定技能图标
        
        # 简化处理：假设找到了优先级技能
        found_priority_skill = False  # 默认未找到
        
        # 如果找到了优先级技能，点击它
        if found_priority_skill:
            x = int(window_info["width"] * 0.5)
            y = int(window_info["height"] * 0.6)
            
            click_success = click_position(
                window_info, x, y,
                config["calibration"]["skill_x"],
                config["calibration"]["skill_y"],
                f"优先级技能: {skill_pattern}"
            )
            
            if click_success:
                time.sleep(config["sleep_times"]["after_skill_selection"])
                # 验证技能界面是否已关闭
                if not detect_skill_selection_screen(window_info):
                    print_info(f"成功选择优先级技能: {skill_pattern}")
                    return True
                else:
                    print_error("技能选择后仍在技能界面，选择失败")
                    break
        else:
            print_debug(f"未找到优先级技能: {skill_pattern}")
    
    # 如果没有找到优先级技能，点击默认位置
    print_info("未找到优先级技能，点击默认位置")
    x = int(window_info["width"] * 0.5)
    y = int(window_info["height"] * 0.6)
    
    click_success = click_position(
        window_info, x, y,
        config["calibration"]["skill_x"],
        config["calibration"]["skill_y"],
        "默认技能位置"
    )
    
    if click_success:
        time.sleep(config["sleep_times"]["after_skill_selection"])
        # 验证技能界面是否已关闭
        if not detect_skill_selection_screen(window_info):
            print_info("技能选择成功，已进入游戏")
            return True
        else:
            print_error("技能选择后仍在技能界面，选择失败")
    
    return False

def get_current_level(window_info):
    """获取当前等级（简化实现）"""
    # 这个函数现在不需要实际实现，因为我们在complete_game_round中使用计数
    # 保留这个函数是为了避免其他地方调用时出错
    return 1  # 临时返回固定值

def click_pause_button(window_info):
    """点击暂停按钮（使用pause.png识别）"""
    max_attempts = config["retry_settings"]["pause_button_max_attempts"]
    
    for attempt in range(max_attempts):
        print_info(f"\n===== 第{attempt+1}/{max_attempts}次尝试：寻找并点击「暂停」按钮 =====")
        
        screenshot = capture_screenshot(window_info)
        if screenshot is None:
            print_error("截图失败，将重试")
            time.sleep(config["sleep_times"]["retry_interval"])
            continue
            
        # 使用pause.png模板识别暂停按钮
        match_result, match_pos, match_val, match_size = multi_template_match(
            screenshot, 
            config["pause_button_templates"]
        )
        
        if match_result and match_pos:
            print_info(f"找到暂停按钮（匹配值：{match_val:.2f}，位置：{match_pos}）")
            
            # 计算点击位置（考虑校准偏移）
            click_x = match_pos[0] + config["calibration"]["pause_x_offset"]
            click_y = match_pos[1] + config["calibration"]["pause_y_offset"]
            
            # 确保点击位置在窗口范围内
            if (click_x < 0 or click_x >= window_info["width"] or 
                click_y < 0 or click_y >= window_info["height"]):
                print_error(f"计算出的点击位置超出窗口范围: ({click_x}, {click_y})")
                print_error(f"窗口尺寸: {window_info['width']}x{window_info['height']}")
                # 使用默认位置
                click_x = int(window_info["width"] * config["pause_button"]["default_x_ratio"])
                click_y = int(window_info["height"] * config["pause_button"]["default_y_ratio"])
                print_info(f"使用默认位置: ({click_x}, {click_y})")
            
            click_success = click_position(
                window_info,
                click_x, click_y,
                0, 0,  # 已经在上面计算了偏移，这里不再添加
                "暂停按钮（图片匹配）"
            )
            
            if click_success:
                time.sleep(config["sleep_times"]["after_pause_click"])
                return True
        else:
            print_info(f"未找到匹配的暂停按钮（最佳匹配值：{match_val:.2f}），尝试默认位置")
            # 点击默认位置
            x = int(window_info["width"] * config["pause_button"]["default_x_ratio"])
            y = int(window_info["height"] * config["pause_button"]["default_y_ratio"])
            click_success = click_position(
                window_info, x, y,
                config["calibration"]["pause_x_offset"],
                config["calibration"]["pause_y_offset"],
                "暂停按钮（默认位置）"
            )
            
            if click_success:
                time.sleep(config["sleep_times"]["after_pause_click"])
                return True
                
        if attempt < max_attempts - 1:
            time.sleep(config["sleep_times"]["retry_interval"])
    
    print_error(f"尝试{max_attempts}次点击暂停按钮失败")
    return False

def detect_pause_menu(window_info):
    """检测暂停菜单是否出现"""
    screenshot = capture_screenshot(window_info)
    if screenshot is None:
        return False
        
    # 检查退出按钮是否存在
    match_result, _, match_val, _ = multi_template_match(
        screenshot, 
        config["exit_button_templates"]
    )
    
    return match_result

def wait_for_pause_menu(window_info):
    """等待暂停菜单出现"""
    print_info(f"\n===== 等待暂停菜单出现（{config['max_wait_seconds']}秒超时，每{config['check_interval']}秒检查一次） =====")
    start_time = time.time()
    
    while time.time() - start_time < config["max_wait_seconds"]:
        if detect_pause_menu(window_info):
            print_info("暂停菜单已出现")
            return True
            
        time.sleep(config["check_interval"])
    
    print_error(f"等待暂停菜单超时（{config['max_wait_seconds']}秒）")
    return False

def click_exit_button(window_info):
    """点击退出按钮"""
    screenshot = capture_screenshot(window_info)
    if screenshot is None:
        return False
        
    match_result, match_pos, match_val, match_size = multi_template_match(
        screenshot, 
        config["exit_button_templates"]
    )
    
    if match_result and match_pos:
        print_info(f"找到退出按钮（匹配值：{match_val:.2f}）")
        click_success = click_position(
            window_info,
            match_pos[0], match_pos[1],
            0, 0,
            "退出按钮（图片匹配）"
        )
        
        if click_success:
            time.sleep(config["sleep_times"]["after_exit_click"])
            return True
    else:
        print_info(f"未找到匹配的退出按钮，尝试默认位置")
        # 点击默认位置
        x = int(window_info["width"] * config["exit_button"]["default_x_ratio"])
        y = int(window_info["height"] * config["exit_button"]["default_y_ratio"])
        click_success = click_position(
            window_info, x, y, 0, 0, "退出按钮（默认位置）"
        )
        
        if click_success:
            time.sleep(config["sleep_times"]["after_exit_click"])
            return True
            
    return False

def detect_confirmation_dialog(window_info):
    """检测确认对话框是否出现"""
    screenshot = capture_screenshot(window_info)
    if screenshot is None:
        return False
        
    # 检查返回按钮是否存在
    match_result, _, match_val, _ = multi_template_match(
        screenshot, 
        config["back_button_templates"]
    )
    
    return match_result

def wait_for_confirmation_dialog(window_info):
    """等待确认对话框出现"""
    print_info(f"\n===== 等待确认对话框出现（{config['max_wait_seconds']}秒超时，每{config['check_interval']}秒检查一次） =====")
    start_time = time.time()
    
    while time.time() - start_time < config["max_wait_seconds"]:
        if detect_confirmation_dialog(window_info):
            print_info("确认对话框已出现")
            return True
            
        time.sleep(config["check_interval"])
    
    print_error(f"等待确认对话框超时（{config['max_wait_seconds']}秒）")
    return False

def click_back_button(window_info):
    """点击返回按钮"""
    screenshot = capture_screenshot(window_info)
    if screenshot is None:
        return False
        
    match_result, match_pos, match_val, match_size = multi_template_match(
        screenshot, 
        config["back_button_templates"]
    )
    
    if match_result and match_pos:
        print_info(f"找到返回按钮（匹配值：{match_val:.2f}）")
        click_success = click_position(
            window_info,
            match_pos[0], match_pos[1],
            0, 0,
            "返回按钮（图片匹配）"
        )
        
        if click_success:
            time.sleep(config["sleep_times"]["after_back_button_click"])
            return True
    else:
        print_info(f"未找到匹配的返回按钮，尝试默认位置")
        # 点击默认位置
        x = int(window_info["width"] * config["back_button"]["default_x_ratio"])
        y = int(window_info["height"] * config["back_button"]["default_y_ratio"])
        click_success = click_position(
            window_info, x, y, 0, 0, "返回按钮（默认位置）"
        )
        
        if click_success:
            time.sleep(config["sleep_times"]["after_back_button_click"])
            return True
            
    return False

def complete_game_round(window_info, target_level):
    """完成一轮游戏"""
    print_info(f"\n===== 开始新一轮游戏，目标等级 {target_level} 级 =====")
    
    # 点击开始游戏
    window_info = click_start_game_with_retry()
    if not window_info:
        return False
    
    # 初始等级为1
    current_level = 1
    print_info(f"初始等级：{current_level}级")
    
    # 循环直到达到目标等级
    while current_level < target_level:
        # 等待进入选择技能界面
        if not wait_for_skill_selection_screen(window_info):
            return False
        
        # 选择技能
        if not select_priority_skill(window_info):
            return False
        
        # 等级增加
        current_level += 1
        print_info(f"选择技能成功，当前等级：{current_level}级")
        
        # 如果不是最后一次升级，等待一段时间
        if current_level < target_level:
            wait_time = config["sleep_times"]["after_skill_selection"]
            print_info(f"等待{wait_time}秒后继续游戏...")
            time.sleep(wait_time)
    
    # 点击暂停按钮
    if not click_pause_button(window_info):
        return False
    
    # 等待暂停菜单出现
    if not wait_for_pause_menu(window_info):
        return False
    
    # 点击退出按钮
    if not click_exit_button(window_info):
        return False
        
    # 等待确认对话框出现
    if not wait_for_confirmation_dialog(window_info):
        return False
        
    # 点击返回按钮
    if not click_back_button(window_info):
        return False
        
    return True

def main_loop():
    """主循环"""
    # 图片读取测试
    test_img_path = get_full_template_path("start.png")
    print_info(f"\n===== 图片读取测试 =====")
    print_info(f"测试路径：{test_img_path}")
    print_info(f"路径是否存在：{os.path.exists(test_img_path)}")
    
    try:
        test_img = cv2.imread(test_img_path)
        if test_img is not None:
            print_info(f"OpenCV读取成功！图片尺寸：{test_img.shape}")
        else:
            print_info(f"OpenCV读取失败！返回None（文件损坏或格式不支持）")
    except Exception as e:
        print_info(f"OpenCV读取时出错：{e}")
    print_info(f"========================\n")
    
    # 检查所有模板是否存在
    missing_templates = check_all_templates()
    if missing_templates:
        print_error("以下模板图片不存在：")
        for path in missing_templates:
            print_error(f"- {path}")
        print_error("请检查模板图片路径是否正确")
        return

    # 检查Tesseract是否可用
    if not check_tesseract(config["tesseract_path"]):
        print_error("Tesseract OCR不可用，程序无法继续运行")
        return

    # 获取游戏窗口信息
    window_info = get_window_info(config["game_title"])
    if not window_info:
        print_error("无法获取游戏窗口信息，程序退出")
        return

    # 激活游戏窗口
    window_info = check_window_foreground(window_info)
    if not window_info:
        print_error("无法将游戏窗口激活到前台，程序退出")
        return

    # 循环执行游戏
    loop_count = config["loop_count"]
    current_loop = 0
    success_count = 0
    fail_count = 0
    
    while loop_count == 0 or current_loop < loop_count:
        current_loop += 1
        print_info(f"\n===== 开始第{current_loop}轮游戏流程 =====")
        
        # 完成一轮游戏
        success = complete_game_round(window_info, config["target_level"])
        
        if success:
            print_info(f"第{current_loop}轮游戏流程完成")
            success_count += 1
        else:
            print_error(f"第{current_loop}轮游戏流程失败")
            fail_count += 1
                
        # 如果不是最后一轮，等待一段时间再开始下一轮
        if loop_count == 0 or current_loop < loop_count:
            wait_time = config["sleep_times"]["after_loop_finish"]
            print_info(f"等待{wait_time}秒后开始下一轮")
            time.sleep(wait_time)

    print_info(f"\n===== 所有游戏流程已完成 =====")
    print_info(f"总轮次：{current_loop}，成功：{success_count}，失败：{fail_count}")

def main():
    """程序入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='游戏自动化脚本')
    parser.add_argument('--play', required=True, help='运行模式，目前仅支持master')
    parser.add_argument('--debug', action='store_true', help='开启调试模式')
    parser.add_argument('--config', default='config.jsonc', help='配置文件路径')
    parser.add_argument('--tesseract-path', help='Tesseract OCR路径（覆盖配置文件）')
    
    args = parser.parse_args()
    
    if args.play != 'master':
        print_error("目前仅支持master模式")
        exit(1)
    
    # 设置全局调试模式
    global DEBUG_MODE
    DEBUG_MODE = args.debug
    
    # 加载配置文件
    load_config(args.config)
    
    # 设置日志
    log_file_path = config.get("log_file_path", "auto_%Y-%m-%d.log")
    actual_log_path = setup_logging(log_file_path)
    print_info(f"日志文件：{actual_log_path}")
    
    print_info(f"开始执行游戏自动化脚本（模式：{args.play}，调试模式：{'开启' if args.debug else '关闭'}）")
    
    # 如果指定了Tesseract路径，覆盖配置文件
    if args.tesseract_path:
        config["tesseract_path"] = args.tesseract_path
    
    # 启动主循环
    try:
        main_loop()
    except Exception as e:
        print_error(f"脚本执行异常：{e}")
        import traceback
        print_error(traceback.format_exc())

if __name__ == "__main__":
    main()