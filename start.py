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
        "max_wait_seconds", "target_level", "loop_count", "sleep_times", "retry_settings", "calibration", "image_processing",
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
    
    # 最终匹配结果 - 确保返回的是标量值而不是数组
    if best_val >= threshold and best_pos is not None:
        # 确保 best_pos 是元组而不是数组
        if hasattr(best_pos, 'any'):  # 如果是 numpy 数组
            best_pos = tuple(best_pos)
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
        
        # 确保 x 和 y 是标量值，不是数组
        if hasattr(x, 'any'):  # 如果是 numpy 数组
            x = x.item() if x.size == 1 else int(x)
        if hasattr(y, 'any'):  # 如果是 numpy 数组
            y = y.item() if y.size == 1 else int(y)
        
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

def detect_in_game_ui(window_info):
    """检测游戏内UI元素，判断是否已经在游戏中"""
    screenshot = capture_screenshot(window_info)
    if screenshot is None:
        return False
        
    # 检测游戏内常见的UI元素（如暂停按钮、技能栏等）
    pause_match, _, pause_val, _ = multi_template_match(
        screenshot, 
        config["pause_button_templates"],
        threshold=0.6
    )
    
    # 检测是否有经验条、血量条等游戏内元素
    # 这里可以根据实际游戏添加更多的检测条件
    
    in_game = pause_match  # 如果找到暂停按钮，说明已经在游戏中
    
    print_debug(f"游戏内UI检测：{'在游戏中' if in_game else '不在游戏中'}（暂停按钮：{pause_val:.2f}）")
    
    return in_game

def is_skill_selection_screen_still_open(window_info, max_checks=3, check_interval=0.5):
    """检查技能选择界面是否仍然打开（增加重试机制）"""
    for i in range(max_checks):
        screenshot = capture_screenshot(window_info)
        if screenshot is None:
            time.sleep(check_interval)
            continue
            
        # 检查技能界面特征
        skill_match, _, skill_val, _ = multi_template_match(
            screenshot, 
            config["skill_template_paths"],
            threshold=0.6
        )
        
        # 检查开始按钮是否出现（说明回到主界面）
        start_match, _, start_val, _ = multi_template_match(
            screenshot, 
            config["start_button_templates"],
            threshold=0.6
        )
        
        # 技能界面关闭的条件：技能特征不存在 或者 开始按钮出现
        if not skill_match or start_match:
            print_debug(f"技能界面已关闭（第{i+1}次检查）")
            return False
        
        time.sleep(check_interval)
    
    print_debug("技能界面仍然打开")
    return True

def detect_activated_skills_window(window_info):
    """检测是否有已激活技能窗口"""
    screenshot = capture_screenshot(window_info)
    if screenshot is None:
        return False
        
    # 检测已激活技能窗口
    match_result, _, match_val, _ = multi_template_match(
        screenshot, 
        config.get("activated_skills_templates", ["activated_skills.png"]),
        threshold=0.6
    )
    
    print_debug(f"已激活技能窗口检测：{'存在' if match_result else '不存在'}（匹配值：{match_val:.2f}）")
    
    return match_result

def click_activated_skills_window(window_info):
    """点击已激活技能窗口的空白处"""
    print_info("检测到已激活技能窗口，点击空白处关闭")
    
    # 计算点击位置（窗口底部中间）
    x = int(window_info["width"] * config.get("activated_skills_click", {}).get("x_ratio", 0.5))
    y = int(window_info["height"] * config.get("activated_skills_click", {}).get("y_ratio", 0.9))
    
    click_success = click_position(
        window_info, x, y,
        0, 0,
        "已激活技能窗口空白处"
    )
    
    if click_success:
        time.sleep(1)  # 等待窗口关闭
        return True
    else:
        print_error("点击已激活技能窗口失败")
        return False

def click_pause_button_with_skill_check(window_info):
    """点击暂停按钮，并处理可能出现的技能选择界面"""
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
                
                # 检查是否出现了技能选择界面
                if detect_skill_selection_screen(window_info):
                    print_info("点击暂停后出现了技能选择界面，先处理技能选择")
                    # 选择技能
                    if select_priority_skill(window_info, False):  # 不是第一次选择
                        print_info("技能选择完成，继续尝试暂停")
                        # 继续尝试暂停
                        continue
                    else:
                        print_error("技能选择失败")
                        return False
                
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
                
                # 检查是否出现了技能选择界面
                if detect_skill_selection_screen(window_info):
                    print_info("点击暂停后出现了技能选择界面，先处理技能选择")
                    # 选择技能
                    if select_priority_skill(window_info, False):  # 不是第一次选择
                        print_info("技能选择完成，继续尝试暂停")
                        # 继续尝试暂停
                        continue
                    else:
                        print_error("技能选择失败")
                        return False
                
                return True
                
        if attempt < max_attempts - 1:
            time.sleep(config["sleep_times"]["retry_interval"])
    
    print_error(f"尝试{max_attempts}次点击暂停按钮失败")
    return False

def detect_skill_selection_screen(window_info):
    """检测是否进入选择技能界面 - 增加更严格的检测条件"""
    screenshot = capture_screenshot(window_info)
    if screenshot is None:
        return False
        
    # 正向匹配：检测技能界面特征
    skill_match, _, skill_val, _ = multi_template_match(
        screenshot, 
        config["skill_template_paths"]
    )
    
    # 反向验证：检查开始按钮是否消失
    start_match, _, start_val, _ = multi_template_match(
        screenshot, 
        config["start_button_templates"],
        threshold=0.6
    )
    
    # 检查是否有其他游戏内UI元素（如暂停按钮）
    pause_match, _, pause_val, _ = multi_template_match(
        screenshot, 
        config["pause_button_templates"],
        threshold=0.6
    )
    
    # 更严格的条件：只有技能界面特征存在且开始按钮不存在且暂停按钮不存在，才判定为技能界面
    is_skill_screen = skill_match and not start_match and not pause_match
    
    # 只在调试模式下输出详细匹配信息
    print_debug(f"技能界面检测：{'存在' if is_skill_screen else '不存在'}（技能：{skill_val:.2f}，开始按钮：{start_val:.2f}，暂停按钮：{pause_val:.2f}）")
    
    return is_skill_screen

def wait_for_skill_selection_screen(window_info):
    """等待进入选择技能界面 - 增加更严格的检测和更长的等待时间"""
    print_info(f"等待进入「选择技能」界面（最多等待{config['max_wait_seconds']}秒）")
    start_time = time.time()
    
    # 需要连续检测到技能界面才认为是真正的技能界面
    consecutive_detections = 0
    required_consecutive = 2  # 需要连续2次检测到技能界面
    
    while time.time() - start_time < config["max_wait_seconds"]:
        if detect_skill_selection_screen(window_info):
            consecutive_detections += 1
            print_info(f"检测到技能界面 ({consecutive_detections}/{required_consecutive})")
            
            if consecutive_detections >= required_consecutive:
                print_info("已进入「选择技能」界面")
                return True
        else:
            consecutive_detections = 0  # 重置连续检测计数
            
        # 检查是否超时
        elapsed = int(time.time() - start_time)
        remaining = int(config["max_wait_seconds"] - elapsed)
        if remaining % 5 == 0:  # 每5秒输出一次等待信息
            print_info(f"等待技能界面... 剩余{remaining}秒")
            
        time.sleep(1)  # 增加检测间隔
    
    print_error(f"等待「选择技能」界面超时（{config['max_wait_seconds']}秒）")
    return False

def detect_hq_screen(window_info):
    """检测是否在寰球界面（通过邀请按钮判断）"""
    screenshot = capture_screenshot(window_info)
    if screenshot is None:
        return False
        
    # 检测邀请按钮是否存在，使用配置的模板
    match_result, _, match_val, _ = multi_template_match(
        screenshot, 
        config.get("hq_invite_templates", ["invite.png", "invite2.png"]),
        threshold=0.6
    )
    
    is_hq_screen = match_result
    print_info(f"寰球界面检测：{'存在' if is_hq_screen else '不存在'}（匹配值：{match_val:.4f}）")
    
    return is_hq_screen

def hq_countdown_timer(window_info, seconds=10):
    """寰球模式倒计时，返回是否可以开始游戏 - 改进判断逻辑"""
    print_info(f"开始{seconds}秒倒计时，等待组队...")
    
    start_time = time.time()
    last_remaining = seconds
    
    while time.time() - start_time < seconds:
        remaining = int(seconds - (time.time() - start_time))
        
        if remaining != last_remaining:
            print_info(f"倒计时剩余：{remaining}秒")
            last_remaining = remaining
        
        # 检查状态
        screenshot = capture_screenshot(window_info)
        if screenshot is None:
            time.sleep(1)
            continue
            
        # 首先检查是否可以开始游戏（这是最重要的状态）
        if detect_hq_start_button(window_info):
            print_info("检测到可以开始游戏（有人加入）")
            
            # 额外检查：确保组队邀请界面已经消失
            if not detect_team_up_interface(window_info):
                print_info("组队邀请界面已消失，确认组队成功")
                # 等待1秒让界面稳定
                print_info("等待1秒让界面稳定...")
                time.sleep(1)
                return True
            else:
                print_info("检测到开始游戏按钮，但组队邀请界面仍然存在，可能是误判")
        
        # 检查是否还在组队邀请界面
        still_in_team_up = detect_team_up_interface(window_info)
        if not still_in_team_up:
            print_info("已退出组队邀请界面（可能是有人加入自动开始游戏）")
            
            # 额外检查：确保可以开始游戏
            if detect_hq_start_button(window_info):
                print_info("退出邀请界面后检测到可以开始游戏，确认组队成功")
                # 等待1秒让界面稳定
                print_info("等待1秒让界面稳定...")
                time.sleep(1)
                return True
            else:
                print_info("退出邀请界面但无法开始游戏，可能返回了寰球界面")
                return False
        
        time.sleep(1)
    
    print_info("倒计时结束")
    
    # 倒计时结束后检查最终状态
    # 检查是否还在邀请界面
    if detect_team_up_interface(window_info):
        print_info("倒计时结束后仍在组队邀请界面（无人加入）")
        return False
    
    # 检查是否可以开始游戏
    if detect_hq_start_button(window_info):
        print_info("倒计时结束后检测到可以开始游戏")
        # 等待1秒让界面稳定
        print_info("等待1秒让界面稳定...")
        time.sleep(1)
        return True
    
    print_info("倒计时结束后已退出邀请界面，但无法开始游戏")
    return False

def detect_hq_start_button(window_info):
    """检测寰球开始游戏按钮 - 增加更严格的检测"""
    screenshot = capture_screenshot(window_info)
    if screenshot is None:
        return False
        
    # 检测开始游戏按钮，使用配置的模板
    start_match, _, start_val, _ = multi_template_match(
        screenshot, 
        config.get("hq_start_templates", ["hq_start.png", "hq_start2.png"]),
        threshold=0.6
    )
    
    # 同时检测组队邀请界面是否消失
    team_up_match = detect_team_up_interface(window_info)
    
    # 只有开始按钮存在且组队邀请界面不存在，才认为是真正的可以开始游戏
    can_start = start_match and not team_up_match
    
    print_info(f"寰球开始游戏检测：{'可以开始' if can_start else '不能开始'}（开始按钮：{start_val:.4f}，组队界面：{'存在' if team_up_match else '不存在'}）")
    
    return can_start

def detect_hq_skill_selection_screen(window_info):
    """检测寰球模式下的技能选择界面 - 改进检测逻辑"""
    screenshot = capture_screenshot(window_info)
    if screenshot is None:
        return False
        
    # 使用寰球模式专用的技能界面模板
    hq_skill_match, _, hq_skill_val, _ = multi_template_match(
        screenshot, 
        config.get("hq_skill_template_paths", ["hq_skill.png", "hq_skill2.png"]),
        threshold=0.7  # 提高阈值，减少误判
    )
    
    # 反向验证：检查开始按钮是否消失
    start_match, _, start_val, _ = multi_template_match(
        screenshot, 
        config["start_button_templates"],
        threshold=0.6
    )
    
    # 对于寰球模式，暂停按钮可能存在，所以不将其作为否定条件
    # 主要条件是：寰球技能界面特征存在且开始按钮不存在
    is_hq_skill_screen = hq_skill_match and not start_match
    
    # 输出详细匹配信息
    print_info(f"寰球技能界面检测：{'存在' if is_hq_skill_screen else '不存在'}（寰球技能：{hq_skill_val:.4f}，开始按钮：{start_val:.4f}）")
    
    return is_hq_skill_screen

def detect_priority_skill(window_info, skill_pattern):
    """检测特定优先级技能是否存在"""
    screenshot = capture_screenshot(window_info)
    if screenshot is None:
        return False, None, None, None
    
    # 定义技能出现的区域（可以根据实际情况调整）
    skill_region = (
        int(window_info["width"] * 0.2),   # x起始
        int(window_info["height"] * 0.3),  # y起始
        int(window_info["width"] * 0.6),   # 宽度
        int(window_info["height"] * 0.5)   # 高度
    )
    
    # 在指定区域内匹配技能
    match_result, match_pos, match_val, match_size = multi_template_match(
        screenshot, 
        [skill_pattern],
        threshold=0.7,  # 提高阈值，减少误判
        region=skill_region
    )
    
    return match_result, match_pos, match_val, match_size

def wait_for_hq_skill_selection_screen(window_info):
    """等待进入寰球模式的选择技能界面 - 增加更严格的检测和超时处理"""
    print_info(f"等待进入「寰球技能选择」界面（最多等待{config['max_wait_seconds']}秒）")
    start_time = time.time()
    
    # 需要连续检测到技能界面才认为是真正的技能界面
    consecutive_detections = 0
    required_consecutive = 2  # 需要连续2次检测到技能界面
    
    # 记录上次检测到技能界面的时间
    last_detection_time = 0
    
    while time.time() - start_time < config["max_wait_seconds"]:
        current_time = time.time()
        
        # 控制检测频率，避免过于频繁
        if current_time - last_detection_time < 0.5:
            time.sleep(0.5 - (current_time - last_detection_time))
            continue
            
        last_detection_time = current_time
        
        if detect_hq_skill_selection_screen(window_info):
            consecutive_detections += 1
            print_info(f"检测到寰球技能界面 ({consecutive_detections}/{required_consecutive})")
            
            if consecutive_detections >= required_consecutive:
                print_info("已进入「寰球技能选择」界面")
                return True
        else:
            consecutive_detections = 0  # 重置连续检测计数
            
        # 检查是否超时
        elapsed = int(time.time() - start_time)
        remaining = int(config["max_wait_seconds"] - elapsed)
        if remaining % 5 == 0:  # 每5秒输出一次等待信息
            print_info(f"等待寰球技能界面... 剩余{remaining}秒")
            
        time.sleep(0.5)  # 检测间隔
    
    print_error(f"等待「寰球技能选择」界面超时（{config['max_wait_seconds']}秒）")
    return False

# 修改 wait_for_skill_selection_screen_to_close 函数，增加更严格的检测
def wait_for_skill_selection_screen_to_close(window_info, max_checks=60, check_interval=0.5):
    """等待技能选择界面关闭 - 增加更严格的检测和超时处理"""
    print_info("等待技能选择界面关闭...")
    
    # 增加初始等待时间，避免过早检测
    time.sleep(2.0)
    
    # 需要连续检测到界面关闭才认为是真正的关闭
    consecutive_closed = 0
    required_consecutive = 3  # 需要连续3次检测到界面关闭
    
    for i in range(max_checks):
        screenshot = capture_screenshot(window_info)
        if screenshot is None:
            time.sleep(check_interval)
            continue
            
        # 检查技能界面特征
        skill_match, _, skill_val, _ = multi_template_match(
            screenshot, 
            config["skill_template_paths"],
            threshold=0.6
        )
        
        # 检查开始按钮是否出现（说明回到主界面）
        start_match, _, start_val, _ = multi_template_match(
            screenshot, 
            config["start_button_templates"],
            threshold=0.6
        )
        
        # 检查是否在游戏中（通过检测游戏内UI元素）
        in_game = detect_in_game_ui(window_info)
        
        # 技能界面关闭的条件：技能特征不存在 或者 开始按钮出现 或者 检测到游戏内UI
        is_closed = not skill_match or start_match or in_game
        
        if is_closed:
            consecutive_closed += 1
            print_info(f"界面关闭检测 ({consecutive_closed}/{required_consecutive})")
            
            if consecutive_closed >= required_consecutive:
                print_info(f"技能界面已关闭（第{i+1}次检查）")
                # 额外等待一段时间确保界面完全稳定
                time.sleep(1.0)
                return True
        else:
            consecutive_closed = 0  # 重置连续检测计数
        
        print_info(f"技能界面仍然存在，等待{check_interval}秒后再次检查...")
        time.sleep(check_interval)
    
    # 即使界面仍然存在，也返回True继续流程，避免因短暂延迟导致失败
    return True

def detect_pause_menu(window_info):
    """检测暂停菜单是否出现 - 增加更严格的验证"""
    screenshot = capture_screenshot(window_info)
    if screenshot is None:
        return False
        
    # 检查退出按钮是否存在
    exit_match, _, exit_val, _ = multi_template_match(
        screenshot, 
        config["exit_button_templates"],
        threshold=0.75  # 提高阈值，减少误判
    )
    
    # 同时检查暂停菜单的其他特征（如果有的话）
    # 例如检查是否有"暂停"文字或其他暂停菜单特有的元素
    
    # 需要同时满足多个条件才认为是真正的暂停菜单
    is_pause_menu = exit_match
    
    # 只在调试模式下输出详细匹配信息
    print_debug(f"暂停菜单检测：{'存在' if is_pause_menu else '不存在'}（退出按钮：{exit_val:.4f}）")
    
    return is_pause_menu

def detect_confirmation_dialog(window_info):
    """检测确认对话框是否出现 - 增加更严格的验证"""
    screenshot = capture_screenshot(window_info)
    if screenshot is None:
        return False
        
    # 检查返回按钮是否存在
    back_match, _, back_val, _ = multi_template_match(
        screenshot, 
        config["back_button_templates"],
        threshold=0.75  # 提高阈值，减少误判
    )
    
    # 同时检查确认对话框的其他特征（如果有的话）
    # 例如检查是否有确认文字或其他对话框特有的元素
    
    # 需要同时满足多个条件才认为是真正的确认对话框
    is_confirmation_dialog = back_match
    
    # 只在调试模式下输出详细匹配信息
    print_debug(f"确认对话框检测：{'存在' if is_confirmation_dialog else '不存在'}（返回按钮：{back_val:.4f}）")
    
    return is_confirmation_dialog

def wait_for_pause_menu(window_info):
    """等待暂停菜单出现 - 增加更严格的验证和重试机制"""
    print_info(f"\n===== 等待暂停菜单出现（{config['max_wait_seconds']}秒超时，每{config['check_interval']}秒检查一次） =====")
    start_time = time.time()
    
    # 需要连续检测到暂停菜单才认为是真正的暂停菜单
    consecutive_detections = 0
    required_consecutive = 2  # 需要连续2次检测到暂停菜单
    
    while time.time() - start_time < config["max_wait_seconds"]:
        if detect_pause_menu(window_info):
            consecutive_detections += 1
            print_info(f"检测到暂停菜单 ({consecutive_detections}/{required_consecutive})")
            
            if consecutive_detections >= required_consecutive:
                print_info("暂停菜单已出现")
                return True
        else:
            consecutive_detections = 0  # 重置连续检测计数
            
        # 检查是否超时
        elapsed = int(time.time() - start_time)
        remaining = int(config["max_wait_seconds"] - elapsed)
        if remaining % 5 == 0:  # 每5秒输出一次等待信息
            print_info(f"等待暂停菜单... 剩余{remaining}秒")
            
        time.sleep(config["check_interval"])
    
    print_error(f"等待暂停菜单超时（{config['max_wait_seconds']}秒）")
    return False

def wait_for_confirmation_dialog(window_info):
    """等待确认对话框出现 - 增加更严格的验证和重试机制"""
    print_info(f"\n===== 等待确认对话框出现（{config['max_wait_seconds']}秒超时，每{config['check_interval']}秒检查一次） =====")
    start_time = time.time()
    
    # 需要连续检测到确认对话框才认为是真正的确认对话框
    consecutive_detections = 0
    required_consecutive = 2  # 需要连续2次检测到确认对话框
    
    while time.time() - start_time < config["max_wait_seconds"]:
        if detect_confirmation_dialog(window_info):
            consecutive_detections += 1
            print_info(f"检测到确认对话框 ({consecutive_detections}/{required_consecutive})")
            
            if consecutive_detections >= required_consecutive:
                print_info("确认对话框已出现")
                return True
        else:
            consecutive_detections = 0  # 重置连续检测计数
            
        # 检查是否超时
        elapsed = int(time.time() - start_time)
        remaining = int(config["max_wait_seconds"] - elapsed)
        if remaining % 5 == 0:  # 每5秒输出一次等待信息
            print_info(f"等待确认对话框... 剩余{remaining}秒")
            
        time.sleep(config["check_interval"])
    
    print_error(f"等待确认对话框超时（{config['max_wait_seconds']}秒）")
    return False

def detect_elite_drop_window(window_info):
    """检测精英掉落界面"""
    screenshot = capture_screenshot(window_info)
    if screenshot is None:
        return False
        
    # 检测精英掉落界面
    match_result, _, match_val, _ = multi_template_match(
        screenshot, 
        config.get("hq_lunpan_templates", ["hq_lunpan.png"]),
        threshold=0.7  # 提高阈值，减少误判
    )
    
    print_info(f"精英掉落界面检测：{'存在' if match_result else '不存在'}（匹配值：{match_val:.4f}）")
    
    return match_result

def click_elite_drop_window(window_info):
    """点击精英掉落界面的空白处"""
    print_info("检测到精英掉落界面，点击空白处关闭")
    
    # 计算点击位置（窗口底部中间）
    x = int(window_info["width"] * config.get("elite_drop_click", {}).get("x_ratio", 0.5))
    y = int(window_info["height"] * config.get("elite_drop_click", {}).get("y_ratio", 0.9))
    
    click_success = click_position(
        window_info, x, y,
        0, 0,
        "精英掉落界面空白处"
    )
    
    if click_success:
        time.sleep(1)  # 等待窗口关闭
        return True
    else:
        print_error("点击精英掉落界面失败")
        return False

def handle_special_interfaces(window_info):
    """处理特殊界面（已激活技能界面、精英掉落界面）"""
    # 等待1秒让界面稳定
    time.sleep(1.0)
    
    # 检查并处理已激活技能界面
    if detect_activated_skills_window(window_info):
        print_info("检测到已激活技能界面，正在处理...")
        if click_activated_skills_window(window_info):
            print_info("已激活技能界面处理完成")
            return True
        else:
            print_error("已激活技能界面处理失败")
            return False
    
    # 检查并处理精英掉落界面
    if detect_elite_drop_window(window_info):
        print_info("检测到精英掉落界面，正在处理...")
        if click_elite_drop_window(window_info):
            print_info("精英掉落界面处理完成")
            return True
        else:
            print_error("精英掉落界面处理失败")
            return False
    
    return False

def click_hq_start_button(window_info):
    """点击寰球开始游戏按钮 - 增加点击前的等待时间"""
    max_attempts = 3
    
    # 获取组队成功后的等待时间（可配置）
    wait_time = config.get("hq_settings", {}).get("after_team_up_wait_time", 1.0)
    
    for attempt in range(max_attempts):
        print_info(f"第{attempt+1}/{max_attempts}次尝试：寻找并点击「开始游戏」按钮")
        
        screenshot = capture_screenshot(window_info)
        if screenshot is None:
            time.sleep(1)
            continue
            
        # 匹配开始游戏按钮
        match_result, match_pos, match_val, match_size = multi_template_match(
            screenshot, 
            config.get("hq_start_templates", ["hq_start.png", "hq_start2.png"]),
            threshold=0.6
        )
        
        if match_result and match_pos:
            print_info(f"找到开始游戏按钮（匹配值：{match_val:.4f}，位置：{match_pos}）")
            
            # 等待配置的时间让界面稳定
            print_info(f"等待{wait_time}秒让界面稳定...")
            time.sleep(wait_time)
            
            click_success = click_position(
                window_info, 
                match_pos[0], match_pos[1], 
                0, 0,
                "开始游戏按钮"
            )
            if click_success:
                # 点击后等待一段时间
                time.sleep(2)
                
                # 检查是否存在特殊界面（已激活技能界面、精英掉落界面）
                if handle_special_interfaces(window_info):
                    print_info("特殊界面处理完成")
                
                return True
        else:
            print_info("未找到开始游戏按钮")
            time.sleep(1)
                
    print_error("无法找到开始游戏按钮")
    return False

def verify_exit_success(window_info):
    """验证退出是否成功（检查是否返回主界面或寰球界面）"""
    max_checks = 5
    check_interval = 2.0
    
    for i in range(max_checks):
        screenshot = capture_screenshot(window_info)
        if screenshot is None:
            time.sleep(check_interval)
            continue
            
        # 检查开始按钮是否出现（说明回到普通模式主界面）
        start_match, _, start_val, _ = multi_template_match(
            screenshot, 
            config["start_button_templates"],
            threshold=0.7  # 提高阈值，减少误判
        )
        
        # 检查暂停按钮是否消失（说明不在游戏中）
        pause_match, _, pause_val, _ = multi_template_match(
            screenshot, 
            config["pause_button_templates"],
            threshold=0.7  # 提高阈值，减少误判
        )
        
        # 检查返回按钮是否消失（说明确认对话框已关闭）
        back_match, _, back_val, _ = multi_template_match(
            screenshot, 
            config["back_button_templates"],
            threshold=0.7  # 提高阈值，减少误判
        )
        
        # 检查邀请组队按钮是否出现（说明回到寰球界面）
        invite_match, _, invite_val, _ = multi_template_match(
            screenshot, 
            config.get("hq_invite_templates", ["invite.png", "invite2.png"]),
            threshold=0.7  # 提高阈值，减少误判
        )
        
        # 退出成功的条件：
        # 1. 开始按钮存在 并且 暂停按钮不存在 并且 返回按钮不存在（普通模式）
        # 2. 或者 邀请组队按钮存在 并且 暂停按钮不存在 并且 返回按钮不存在（寰球模式）
        exit_success = (
            (start_match and not pause_match and not back_match) or
            (invite_match and not pause_match and not back_match)
        )
        
        if exit_success:
            if start_match:
                print_info("退出成功，已返回普通模式主界面")
            elif invite_match:
                print_info("退出成功，已返回寰球界面")
            return True
        
        # 如果返回按钮仍然存在，说明退出失败
        if back_match:
            print_info(f"返回按钮仍然存在（匹配值：{back_val:.4f}），退出可能失败")
        
        # 输出当前状态信息，便于调试
        print_info(f"退出验证状态 - 开始按钮: {start_match}({start_val:.4f}), "
                  f"暂停按钮: {pause_match}({pause_val:.4f}), "
                  f"返回按钮: {back_match}({back_val:.4f}), "
                  f"邀请按钮: {invite_match}({invite_val:.4f})")
        
        print_info(f"等待退出完成（{i+1}/{max_checks}）")
        time.sleep(check_interval)
    
    print_info("退出验证失败，可能未成功返回主界面或寰球界面")
    return False

def click_exit_button_with_retry(window_info, max_attempts=60, retry_interval=0.5):
    """点击退出按钮，如果未出现确认对话框则重试，并处理特殊界面"""
    for attempt in range(max_attempts):
        print_info(f"第{attempt+1}/{max_attempts}次尝试：点击退出按钮")
        
        # 先检查并处理特殊界面
        if handle_special_interfaces_before_action(window_info):
            print_info("处理了特殊界面，继续尝试退出")
            # 处理完特殊界面后，等待一段时间
            time.sleep(1.0)
        
        # 点击退出按钮
        if not click_exit_button(window_info):
            print_error("点击退出按钮失败")
            if attempt < max_attempts - 1:
                time.sleep(retry_interval)
            continue
        
        # 等待一段时间让界面响应
        time.sleep(1.0)
        
        # 检查确认对话框是否出现
        if detect_confirmation_dialog(window_info):
            print_info("确认对话框已出现")
            return True
        
        print_info("确认对话框未出现，将重试")
        if attempt < max_attempts - 1:
            time.sleep(retry_interval)
    
    print_error(f"尝试{max_attempts}次点击退出按钮，但确认对话框未出现")
    return False

def click_back_button_with_verification(window_info, max_attempts=3):
    """点击返回按钮并验证是否成功，处理特殊界面"""
    for attempt in range(max_attempts):
        print_info(f"第{attempt+1}/{max_attempts}次尝试：点击返回按钮")
        
        # 先检查并处理特殊界面
        if handle_special_interfaces_before_action(window_info):
            print_info("处理了特殊界面，继续尝试点击返回按钮")
            # 处理完特殊界面后，等待一段时间
            time.sleep(1.0)
        
        # 点击返回按钮
        if not click_back_button(window_info):
            print_error("点击返回按钮失败")
            if attempt < max_attempts - 1:
                time.sleep(1)
            continue
        
        # 等待一段时间让界面响应
        time.sleep(1.0)
        
        # 检查返回按钮是否消失
        screenshot = capture_screenshot(window_info)
        if screenshot is None:
            print_error("无法获取截图，无法验证返回按钮")
            if attempt < max_attempts - 1:
                time.sleep(1)
            continue
        
        back_match, _, back_val, _ = multi_template_match(
            screenshot, 
            config["back_button_templates"],
            threshold=0.7
        )
        
        if not back_match:
            print_info("返回按钮已消失，点击成功")
            return True
        else:
            print_info(f"返回按钮仍然存在（匹配值：{back_val:.4f}），点击可能失败")
            
            if attempt < max_attempts - 1:
                time.sleep(1)
    
    print_error(f"尝试{max_attempts}次点击返回按钮均失败")
    return False

def click_pause_button_with_retry(window_info, max_attempts=60, retry_interval=0.5):
    """点击暂停按钮，如果未出现暂停菜单则重试，并处理特殊界面"""
    for attempt in range(max_attempts):
        print_info(f"第{attempt+1}/{max_attempts}次尝试：点击暂停按钮")
        
        # 先检查并处理特殊界面
        if handle_special_interfaces_before_action(window_info):
            print_info("处理了特殊界面，继续尝试暂停")
            # 处理完特殊界面后，等待一段时间
            time.sleep(1.0)
        
        # 点击暂停按钮
        if not click_pause_button(window_info):
            print_error("点击暂停按钮失败")
            if attempt < max_attempts - 1:
                time.sleep(retry_interval)
            continue
        
        # 等待一段时间让界面响应
        time.sleep(1.0)
        
        # 检查暂停菜单是否出现
        if detect_pause_menu(window_info):
            print_info("暂停菜单已出现")
            return True
        
        print_info("暂停菜单未出现，检查是否有特殊界面")
        
        # 检查并处理特殊界面
        if handle_special_interfaces_before_action(window_info):
            print_info("处理了特殊界面，继续尝试暂停")
            # 处理完特殊界面后，等待一段时间
            time.sleep(1.0)
            
            # 再次检查暂停菜单是否出现
            if detect_pause_menu(window_info):
                print_info("暂停菜单已出现")
                return True
        
        print_info("暂停菜单仍未出现，将重试")
        if attempt < max_attempts - 1:
            time.sleep(retry_interval)
    
    print_error(f"尝试{max_attempts}次点击暂停按钮，但暂停菜单未出现")
    return False

def handle_special_interfaces_before_action(window_info):
    """在执行任何操作前，先检查并处理特殊界面（已激活技能界面、精英掉落界面、技能选择界面）"""
    # 等待1秒让界面稳定
    time.sleep(1.0)
    
    handled = False
    
    # 检查并处理已激活技能界面
    if detect_activated_skills_window(window_info):
        print_info("检测到已激活技能界面，正在处理...")
        if click_activated_skills_window(window_info):
            print_info("已激活技能界面处理完成")
            handled = True
        else:
            print_error("已激活技能界面处理失败")
    
    # 检查并处理精英掉落界面
    if detect_elite_drop_window(window_info):
        print_info("检测到精英掉落界面，正在处理...")
        if click_elite_drop_window(window_info):
            print_info("精英掉落界面处理完成")
            handled = True
        else:
            print_error("精英掉落界面处理失败")
    
    # 检查并处理技能选择界面
    if detect_skill_selection_screen(window_info):
        print_info("检测到技能选择界面，正在处理...")
        # 选择默认技能
        if select_skill(window_info, False):  # 不是第一次选择
            print_info("技能选择界面处理完成")
            handled = True
        else:
            print_error("技能选择界面处理失败")
    
    return handled

def pause_and_exit_with_retry(window_info, max_attempts=3):
    """带重试机制的暂停和退出流程 - 处理特殊界面"""
    for attempt in range(max_attempts):
        print_info(f"\n===== 第{attempt+1}/{max_attempts}次尝试：暂停并退出游戏 =====")
        
        # 点击暂停按钮（处理特殊界面）
        if not click_pause_button_with_retry(window_info):
            print_error("点击暂停按钮失败")
            continue
        
        # 等待暂停菜单出现
        if not wait_for_pause_menu(window_info):
            print_error("等待暂停菜单超时")
            
            # 检查并处理特殊界面
            if handle_special_interfaces_before_action(window_info):
                print_info("处理了特殊界面，继续尝试暂停")
                continue
            
            continue
        
        # 点击退出按钮（处理特殊界面）
        if not click_exit_button_with_retry(window_info):
            print_error("点击退出按钮失败")
            continue
        
        # 等待确认对话框出现
        if not wait_for_confirmation_dialog(window_info):
            print_error("等待确认对话框超时")
            continue
        
        # 点击返回按钮并验证是否成功（处理特殊界面）
        if not click_back_button_with_verification(window_info):
            print_error("点击返回按钮失败")
            continue
        
        # 验证退出是否成功
        if verify_exit_success(window_info):
            print_info("退出流程完成")
            return True
        else:
            print_error("退出验证失败")
    
    print_error(f"尝试{max_attempts}次暂停退出流程均失败")
    return False

def select_skill(window_info, is_first_selection=False):
    """选择技能 - 点击默认位置"""
    print_info("开始选择技能...")
    
    # 如果是第一次选择技能，检查并处理已激活技能窗口
    if is_first_selection:
        print_info("检查是否第一次选择技能，需要处理已激活技能窗口")
        if detect_activated_skills_window(window_info):
            print_info("检测到已激活技能窗口，正在处理...")
            if not click_activated_skills_window(window_info):
                print_error("处理已激活技能窗口失败")
                return False
            # 等待界面稳定
            time.sleep(2.0)
        else:
            print_info("未检测到已激活技能窗口")
    
    # 增加额外的等待时间，确保技能界面完全加载
    time.sleep(1.5)
    
    # 点击默认位置
    return click_default_skill_position(window_info)

def click_default_skill_position(window_info):
    """点击默认技能位置"""
    print_info("点击默认技能位置")
    x = int(window_info["width"] * 0.5)
    y = int(window_info["height"] * 0.6)
    
    click_success = click_position(
        window_info, x, y,
        config["calibration"]["skill_x"],
        config["calibration"]["skill_y"],
        "默认技能位置"
    )
    
    if click_success:
        # 增加选择后的等待时间，确保界面稳定
        time.sleep(config["sleep_times"]["after_skill_selection"] + 1.0)
        
        # 验证技能界面是否已关闭
        if wait_for_skill_selection_screen_to_close(window_info, max_checks=12, check_interval=1.0):
            print_info("技能选择成功，已进入游戏")
            return True
        else:
            print_error("技能选择后仍在技能界面，选择失败")
    
    return False

def execute_hq_game_flow(window_info, target_level):
    """执行寰球模式的游戏流程 - 使用简化的技能选择"""
    print_info("开始执行寰球模式游戏流程...")
    
    try:
        # 首先检查是否已经在游戏中（可能自动开始了）
        if detect_hq_skill_selection_screen(window_info):
            print_info("检测到已经在技能选择界面，直接开始选择技能")
        else:
            # 尝试检测并点击开始游戏按钮
            time.sleep(2)
            screenshot = capture_screenshot(window_info)
            # 检查截图是否有效（非None且包含数据）
            if screenshot is not None and screenshot.size > 0:
                match_result, match_pos, match_val, match_size = multi_template_match(
                    screenshot, 
                    config.get("hq_start_templates", ["hq_start.png", "hq_start2.png"]),
                    threshold=0.6
                )
                
                if match_result and match_pos is not None:
                    # 确保位置坐标是标量值
                    pos_x, pos_y = match_pos
                    if hasattr(pos_x, 'any'):  # 如果是 numpy 数组
                        pos_x = pos_x.item() if pos_x.size == 1 else int(pos_x)
                    if hasattr(pos_y, 'any'):  # 如果是 numpy 数组
                        pos_y = pos_y.item() if pos_y.size == 1 else int(pos_y)
                        
                    print_info(f"找到寰球开始按钮（匹配值：{match_val:.4f}），准备点击...")
                    
                    # 等待2秒让界面稳定
                    print_info("等待2秒让界面稳定...")
                    time.sleep(2)
                    
                    click_success = click_position(
                        window_info, 
                        pos_x, pos_y, 
                        0, 0,
                        "寰球开始按钮"
                    )
                    if click_success:
                        print_info("成功点击寰球开始游戏按钮")
                        time.sleep(3)
                    else:
                        print_error("点击寰球开始按钮失败")
                        return False
                else:
                    print_info("未找到寰球开始按钮，可能已经自动开始游戏")
        
        # 等待游戏开始并检查是否进入游戏
        print_info("等待进入游戏...")
        if not wait_for_hq_skill_selection_screen(window_info):
            print_error("未能成功进入游戏")
            return False
        
        print_info("成功进入游戏，开始执行升级流程...")
        
        # 执行升级流程 - 通过技能选择次数计算等级
        current_level = 1
        is_first_selection = True  # 标记是否是第一次选择技能
        
        while current_level < target_level:
            print_info(f"等待升级到{current_level + 1}级...")
            
            # 等待技能选择界面 - 使用寰球专用的等待函数
            if not wait_for_hq_skill_selection_screen(window_info):
                print_error("等待寰球技能选择界面超时")
                return False
            
            # 选择技能（如果是第一次选择，先处理已激活技能窗口）
            if not select_skill(window_info, is_first_selection):
                print_error("选择技能失败")
                return False
            
            # 第一次选择完成后，标记为False
            if is_first_selection:
                is_first_selection = False
            
            current_level += 1
            print_info(f"升级成功，当前等级：{current_level}级")
            
            # 增加额外的等待时间，确保界面稳定
            if current_level < target_level:
                wait_time = config["sleep_times"]["after_skill_selection"]
                print_info(f"等待{wait_time}秒后继续游戏...")
                time.sleep(wait_time)
        
        # 完成游戏退出流程
        print_info(f"已达到目标等级{target_level}级，准备退出游戏")
        if pause_and_exit_with_retry(window_info):
            print_info("游戏退出流程完成")
            return True
        else:
            print_error("游戏退出流程失败")
            return False
            
    except Exception as e:
        print_error(f"执行寰球游戏流程时发生异常: {e}")
        import traceback
        print_error(traceback.format_exc())
        return False

def complete_game_round(window_info, target_level):
    """完成一轮游戏 - 使用简化的技能选择"""
    print_info(f"\n===== 开始新一轮游戏，目标等级 {target_level} 级 =====")
    
    # 点击开始游戏
    window_info = click_start_game_with_retry()
    if not window_info:
        return False
    
    # 初始等级为1
    current_level = 1
    print_info(f"初始等级：{current_level}级")
    
    # 循环直到达到目标等级
    is_first_selection = True  # 标记是否是第一次选择技能
    while current_level < target_level:
        # 等待进入选择技能界面
        if not wait_for_skill_selection_screen(window_info):
            return False
        
        # 选择技能（如果是第一次选择，先处理已激活技能窗口）
        if not select_skill(window_info, is_first_selection):
            return False
        
        # 第一次选择完成后，标记为False
        if is_first_selection:
            is_first_selection = False
        
        # 等级增加
        current_level += 1
        print_info(f"选择技能成功，当前等级：{current_level}级")
        
        # 如果不是最后一次升级，等待一段时间
        if current_level < target_level:
            wait_time = config["sleep_times"]["after_skill_selection"]
            print_info(f"等待{wait_time}秒后继续游戏...")
            time.sleep(wait_time)
    
    # 完成游戏退出流程
    print_info(f"已达到目标等级{target_level}级，准备退出游戏")
    if pause_and_exit_with_retry(window_info):
        print_info("游戏退出流程完成")
        return True
    else:
        print_error("游戏退出流程失败")
        return False

def detect_team_up_interface(window_info):
    """检测是否在组队邀请界面 - 检测标题和邀请按钮"""
    screenshot = capture_screenshot(window_info)
    if screenshot is None:
        return False
        
    # 检测组队界面标题
    title_match, _, title_val, _ = multi_template_match(
        screenshot, 
        config.get("hq_team_up_title_templates", ["hq_invite_title.png"]),
        threshold=0.6
    )
    
    # 检测邀请按钮 - 降低阈值以提高识别率
    invite_match, _, invite_val, _ = multi_template_match(
        screenshot, 
        config.get("hq_invite_templates", ["invite.png", "invite2.png"]),
        threshold=0.5  # 降低阈值从0.6到0.5
    )
    
    # 需要同时检测到标题和邀请按钮才认为是真正的组队邀请界面
    is_team_up = title_match and invite_match
    
    print_info(f"组队邀请界面检测：{'存在' if is_team_up else '不存在'}（标题：{title_val:.4f}，邀请按钮：{invite_val:.4f}）")
    
    # 如果标题匹配成功但按钮匹配失败，输出调试信息
    if title_match and not invite_match:
        print_info("标题匹配成功但按钮匹配失败，可能需要更新按钮模板或调整阈值")
        
    return is_team_up

def click_invite_button(window_info):
    """点击邀请按钮 - 降低阈值以提高识别率"""
    max_attempts = 3
    
    for attempt in range(max_attempts):
        print_info(f"第{attempt+1}/{max_attempts}次尝试：寻找并点击「邀请」按钮")
        
        screenshot = capture_screenshot(window_info)
        if screenshot is None:
            time.sleep(1)
            continue
            
        # 只匹配邀请按钮，不匹配标题 - 降低阈值
        match_result, match_pos, match_val, match_size = multi_template_match(
            screenshot, 
            config.get("hq_invite_templates", ["invite.png", "invite2.png"]),
            threshold=0.5  # 降低阈值从0.6到0.5
        )
        
        if match_result and match_pos:
            print_info(f"找到邀请按钮（匹配值：{match_val:.4f}，位置：{match_pos}）")
            
            # 等待0.5秒让界面稳定
            print_info("等待0.5秒让界面稳定...")
            time.sleep(0.5)
            
            click_success = click_position(
                window_info, 
                match_pos[0], match_pos[1], 
                0, 0,
                "邀请按钮"
            )
            if click_success:
                time.sleep(1)
                return True
        else:
            print_info("未找到邀请按钮，尝试使用默认位置")
            # 如果模板匹配失败，尝试使用默认位置
            if click_invite_button_default(window_info):
                return True
            time.sleep(1)
                
    print_error("无法找到邀请按钮")
    return False

def click_invite_button_default(window_info):
    """点击邀请按钮的默认位置"""
    print_info("尝试点击邀请按钮的默认位置")
    
    # 计算默认位置（可以根据实际情况调整）
    x = int(window_info["width"] * config.get("invite_button", {}).get("default_x_ratio", 0.7))
    y = int(window_info["height"] * config.get("invite_button", {}).get("default_y_ratio", 0.8))
    
    click_success = click_position(
        window_info, 
        x, y,
        0, 0,
        "邀请按钮（默认位置）"
    )
    
    if click_success:
        time.sleep(1)
        return True
    
    return False

def complete_hq_round(window_info, target_level):
    """完成一轮寰球模式游戏 - 修复招募流程逻辑"""
    print_info(f"\n===== 开始寰球模式游戏，目标等级 {target_level} 级 =====")
    
    max_recruit_attempts = config.get("hq_settings", {}).get("max_recruit_attempts", 10)
    recruit_count = 0
    
    while recruit_count < max_recruit_attempts:
        recruit_count += 1
        print_info(f"\n--- 第{recruit_count}/{max_recruit_attempts}次招募尝试 ---")
        
        # 1. 检测当前是否在寰球界面或组队邀请界面
        in_hq_screen = detect_hq_screen(window_info)
        in_team_up_interface = detect_team_up_interface(window_info)
        
        if not in_hq_screen and not in_team_up_interface:
            print_error("当前不在寰球界面也不在组队邀请界面")
            return False
        
        # 2. 如果在组队邀请界面，直接点击发布招募按钮
        if in_team_up_interface:
            print_info("已经在组队邀请界面，直接点击发布招募按钮")
            if not click_send_invite_button(window_info):
                print_error("点击发布招募按钮失败")
                return False
        else:
            # 3. 如果在寰球界面，先点击邀请按钮进入组队邀请界面
            print_info("当前在寰球界面，点击邀请按钮进入组队邀请界面")
            if not click_invite_button(window_info):
                print_error("点击邀请按钮失败")
                return False
            
            # 4. 等待组队邀请界面出现
            team_up_detected = False
            wait_time = config.get("hq_settings", {}).get("team_up_wait_time", 10)
            for i in range(wait_time):
                if detect_team_up_interface(window_info):
                    team_up_detected = True
                    break
                time.sleep(1)
                print_info(f"等待组队界面出现... ({i+1}/{wait_time})")
            
            if not team_up_detected:
                print_error("未进入组队邀请界面")
                return False
            
            print_info("成功进入组队邀请界面")
            
            # 5. 点击发布招募按钮
            if not click_send_invite_button(window_info):
                print_error("点击发布招募按钮失败")
                return False
        
        print_info("已发布招募，开始倒计时等待组队...")
        
        # 6. 倒计时等待组队
        countdown_seconds = config.get("hq_settings", {}).get("countdown_seconds", 10)
        can_start = hq_countdown_timer(window_info, countdown_seconds)
        
        if can_start:
            print_info("组队成功，可以开始游戏")
            
            # 7. 点击开始游戏按钮
            if not click_hq_start_button(window_info):
                print_error("点击开始游戏按钮失败")
                return False
            
            # 8. 执行开始游戏和游戏流程
            game_success = execute_hq_game_flow(window_info, target_level)
            if game_success:
                return True
            else:
                print_error("游戏流程执行失败，结束流程")
                return False  # 游戏流程失败，直接结束
        else:
            print_info(f"第{recruit_count}次招募未成功组队，继续尝试...")
            # 继续下一次招募尝试
    
    # 如果达到最大尝试次数仍未成功
    if recruit_count >= max_recruit_attempts:
        print_error(f"已达到最大招募尝试次数({max_recruit_attempts})，组队失败")
        return False
    
    return False

def click_send_invite_button(window_info):
    """点击发布招募按钮 - 增加点击前的等待时间"""
    max_attempts = 3
    
    for attempt in range(max_attempts):
        print_info(f"第{attempt+1}/{max_attempts}次尝试：寻找并点击「发布招募」按钮")
        
        screenshot = capture_screenshot(window_info)
        if screenshot is None:
            time.sleep(1)
            continue
            
        # 匹配发布招募按钮，使用配置的模板
        match_result, match_pos, match_val, match_size = multi_template_match(
            screenshot, 
            config.get("hq_send_invite_templates", ["send_invite.png", "send_invite2.png"]),
            threshold=0.6
        )
        
        if match_result and match_pos:
            print_info(f"找到发布招募按钮（匹配值：{match_val:.4f}，位置：{match_pos}）")
            
            # 等待0.5秒让界面稳定
            print_info("等待0.5秒让界面稳定...")
            time.sleep(0.5)
            
            click_success = click_position(
                window_info, 
                match_pos[0], match_pos[1], 
                0, 0,
                "发布招募按钮"
            )
            if click_success:
                return True
        else:
            print_info("未找到发布招募按钮，尝试使用默认位置")
            # 如果模板匹配失败，尝试使用默认位置
            if click_send_invite_button_default(window_info):
                return True
            time.sleep(1)
                
    print_error("无法找到发布招募按钮")
    return False

def click_send_invite_button_default(window_info):
    """点击发布招募按钮的默认位置"""
    print_info("尝试点击发布招募按钮的默认位置")
    
    # 计算默认位置（可以根据实际情况调整）
    x = int(window_info["width"] * config.get("send_invite_button", {}).get("default_x_ratio", 0.5))
    y = int(window_info["height"] * config.get("send_invite_button", {}).get("default_y_ratio", 0.7))
    
    click_success = click_position(
        window_info, 
        x, y,
        0, 0,
        "发布招募按钮（默认位置）"
    )
    
    if click_success:
        return True
    
    return False

def check_tesseract(tesseract_path):
    """检查Tesseract是否可用"""
    try:
        # 设置Tesseract路径
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        # 测试Tesseract是否正常工作
        test_img = np.zeros((100, 100), np.uint8)
        cv2.putText(test_img, "Test", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # 尝试简单的OCR识别
        text = pytesseract.image_to_string(test_img)
        print_info(f"Tesseract OCR测试成功: {text.strip()}")
        return True
    except Exception as e:
        print_error(f"Tesseract OCR检查失败：{e}")
        
        # 检查是否是路径问题
        if not os.path.exists(tesseract_path):
            print_error(f"Tesseract路径不存在: {tesseract_path}")
        else:
            print_error("Tesseract路径存在，但无法正常工作")
            
        return False

def detect_priority_skill_by_text(window_info, skill_names):
    """通过文字识别检测特定优先级技能"""
    screenshot = capture_screenshot(window_info)
    if screenshot is None:
        return False, None, None
    
    # 定义技能出现的区域（可以根据实际情况调整）
    skill_region = (
        int(window_info["width"] * 0.2),   # x起始
        int(window_info["height"] * 0.3),  # y起始
        int(window_info["width"] * 0.6),   # 宽度
        int(window_info["height"] * 0.5)   # 高度
    )
    
    # 截取技能区域
    x, y, w, h = skill_region
    skill_area = screenshot[y:y+h, x:x+w]
    
    # 预处理图像以提高OCR识别率
    skill_area = preprocess_image(skill_area)
    
    # 使用OCR识别技能文字
    try:
        # 设置Tesseract参数
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(skill_area, config=custom_config)
        print_debug(f"OCR识别结果: {text}")
        
        # 检查是否有优先级技能
        for skill_name in skill_names:
            if skill_name.lower() in text.lower():
                print_info(f"找到优先级技能: {skill_name}")
                
                # 尝试找到技能的位置（简单实现，可以根据需要改进）
                # 这里我们假设技能在区域的中间位置
                skill_pos = (x + w // 2, y + h // 2)
                return True, skill_pos, skill_name
                
    except Exception as e:
        print_error(f"OCR识别失败: {e}")
        
        # 检查是否是Tesseract安装问题
        if "tesseract is not installed" in str(e).lower():
            print_error("Tesseract OCR未安装或未正确配置")
            print_error("请安装Tesseract OCR并将其添加到系统PATH中")
            print_error("或者确保配置文件中指定的Tesseract路径正确")
        
    return False, None, None

def select_priority_skill(window_info, is_first_selection=False):
    """选择优先级最高的技能 - 只使用文字识别"""
    print_info("开始选择技能...")
    
    # 如果是第一次选择技能，检查并处理已激活技能窗口
    if is_first_selection:
        print_info("检查是否第一次选择技能，需要处理已激活技能窗口")
        if detect_activated_skills_window(window_info):
            print_info("检测到已激活技能窗口，正在处理...")
            if not click_activated_skills_window(window_info):
                print_error("处理已激活技能窗口失败")
                return False
            # 等待界面稳定
            time.sleep(2.0)
        else:
            print_info("未检测到已激活技能窗口")
    
    # 增加额外的等待时间，确保技能界面完全加载
    time.sleep(1.5)
    
    # 检查是否配置了优先级技能名称
    priority_skills = config.get("priority_skill_names", [])
    
    # 如果配置了优先级技能名称，尝试使用文字识别
    if priority_skills:
        print_info("尝试使用文字识别选择优先级技能")
        found, pos, skill_name = detect_priority_skill_by_text(window_info, priority_skills)
        
        if found and pos:
            print_info(f"通过文字识别找到优先级技能 {skill_name}，位置：{pos}")
            
            # 计算点击位置
            click_x = pos[0]
            click_y = pos[1]
            
            click_success = click_position(
                window_info, 
                click_x, click_y,
                0, 0,
                f"优先级技能: {skill_name}"
            )
            
            if click_success:
                # 增加选择后的等待时间，确保界面稳定
                time.sleep(config["sleep_times"]["after_skill_selection"])
                
                # 验证技能界面是否已关闭
                if wait_for_skill_selection_screen_to_close(window_info, max_checks=12, check_interval=1.0):
                    print_info(f"成功选择优先级技能: {skill_name}")
                    return True
                else:
                    print_error("技能选择后仍在技能界面，选择失败")
                    return False
        else:
            print_info("文字识别未找到优先级技能，将点击默认位置")
    else:
        print_info("未配置优先级技能名称，将点击默认位置")
    
    # 如果文字识别失败或未配置，点击默认位置
    return click_default_skill_position(window_info)

def main():
    """程序入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='游戏自动化脚本')
    parser.add_argument('--play', required=True, help='运行模式，支持master或hq')
    parser.add_argument('--debug', action='store_true', help='开启调试模式')
    parser.add_argument('--config', default='config.jsonc', help='配置文件路径')
    parser.add_argument('--tesseract-path', help='Tesseract OCR路径（覆盖配置文件）')
    
    args = parser.parse_args()
    
    if args.play not in ['master', 'hq']:
        print_error("目前仅支持master或hq模式")
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
    
    # 检查Tesseract是否可用
    if not check_tesseract(config["tesseract_path"]):
        print_error("Tesseract OCR不可用，程序无法继续运行")
        print_error("请安装Tesseract OCR并将其添加到系统PATH中")
        print_error("或者确保配置文件中指定的Tesseract路径正确")
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
    
    # 根据模式执行不同的逻辑
    if args.play == 'master':
        main_loop()
    elif args.play == 'hq':
        # 寰球模式主循环
        loop_count = config["loop_count"]
        current_loop = 0
        success_count = 0
        fail_count = 0
        
        while loop_count == 0 or current_loop < loop_count:
            current_loop += 1
            print_info(f"\n===== 开始第{current_loop}轮寰球模式流程 =====")
            
            # 完成一轮寰球模式游戏
            success = complete_hq_round(window_info, config["target_level"])
            
            if success:
                print_info(f"第{current_loop}轮寰球模式流程完成")
                success_count += 1
            else:
                print_error(f"第{current_loop}轮寰球模式流程失败")
                fail_count += 1
                    
            # 如果不是最后一轮，等待一段时间再开始下一轮
            if loop_count == 0 or current_loop < loop_count:
                wait_time = config["sleep_times"]["after_loop_finish"]
                print_info(f"等待{wait_time}秒后开始下一轮")
                time.sleep(wait_time)
        
        print_info(f"\n===== 所有寰球模式流程已完成 =====")
        print_info(f"总轮次：{current_loop}，成功：{success_count}，失败：{fail_count}")
    
    # 启动主循环
    try:
        if args.play == 'master':
            main_loop()
    except Exception as e:
        print_error(f"脚本执行异常：{e}")
        import traceback
        print_error(traceback.format_exc())

if __name__ == "__main__":
    main()