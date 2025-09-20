# 游戏自动化脚本

这是一个用于自动化执行《向僵尸开炮》游戏的Python脚本，通过图像识别技术模拟玩家操作，实现自动开始游戏、选择技能、升级和退出游戏的功能。

## 功能特点

- 🎮 自动识别游戏窗口并激活
- 🖱️ 基于模板匹配的按钮识别与点击
- 🔍 多模板支持提高识别率
- ⚡ 可配置的优先级技能选择
- 📊 等级计数与目标达成检测
- 🔧 丰富的配置选项和校准设置
- 📝 详细的日志记录系统

## 技术栈

- **图像识别**: OpenCV + 模板匹配
- **文字识别**: Tesseract OCR (可选)
- **屏幕捕获**: MSS
- **窗口管理**: PyGetWindow
- **鼠标控制**: PyAutoGUI
- **配置管理**: JSON5

## 安装依赖

1. 安装Python 3.7+
2. 安装所需库：
```bash
pip install opencv-python pyautogui pygetwindow mss pytesseract json5 pillow
```
3. 安装Tesseract OCR (可选，用于文字识别):
   - Windows: 下载并安装 [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)
   - 其他系统: 使用包管理器安装

## 配置文件说明

配置文件(`config.jsonc`)包含所有可调整的参数，主要分为以下几类：

### 路径配置
- `template_root`: 模板图片存储目录
- `tesseract_path`: Tesseract OCR引擎路径

### 模板配置
- `skill_template_paths`: 技能界面识别模板
- `start_button_templates`: 开始按钮识别模板
- `exit_button_templates`: 退出按钮识别模板
- `back_button_templates`: 返回按钮识别模板
- `pause_button_templates`: 暂停按钮识别模板

### 游戏设置
- `target_level`: 目标等级
- `loop_count`: 循环次数(0表示无限循环)
- `priority_skill_patterns`: 优先级技能列表

### 识别参数
- `match_threshold`: 图像匹配阈值(0-1)
- `match_method`: 图像匹配算法
- `image_processing`: 图像预处理设置

### 时间控制
- `check_interval`: 检查间隔(秒)
- `max_wait_seconds`: 最大等待时间(秒)
- `sleep_times`: 各种操作的等待时间配置

### 坐标校准
- `calibration`: 各种按钮的点击位置校准
- 按钮默认位置比例设置

## 使用方法

### 基本使用
```bash
python start.py --play master --config config.jsonc
```

### 启用调试模式
```bash
python start.py --play master --config config.jsonc --debug
```

### 指定Tesseract路径
```bash
python start.py --play master --config config.jsonc --tesseract-path "C:\Tesseract-OCR\tesseract.exe"
```

## 模板图片准备

1. 创建模板目录(如`D:\Workspace\xjskp-auto\templates`)
2. 截取游戏中的各种界面元素并保存为PNG格式：
   - 开始按钮(`start.png`)
   - 技能界面(`skill_template.png`, `skill_template2.png`)
   - 暂停按钮(`pause.png`)
   - 退出按钮(`exit_btn.png`, `exit_btn2.png`)
   - 返回按钮(`back_btn.png`, `back_btn2.png`)

## 自定义配置

根据实际游戏情况，您可能需要调整以下配置：

1. **匹配阈值**: 如果识别率低，降低`match_threshold`(如0.6)；如果误识别多，提高阈值(如0.8)
2. **坐标校准**: 如果点击位置不准确，调整`calibration`中的偏移值
3. **等待时间**: 根据游戏响应速度调整`sleep_times`中的各种等待时间
4. **默认位置**: 如果界面布局变化，调整按钮的默认位置比例

## 常见问题

### 窗口无法激活
- 确保游戏窗口标题与配置中的`game_title`匹配
- 检查游戏是否以窗口模式运行

### 识别率低
- 使用更清晰的模板图片
- 调整图像处理参数(对比度、阈值等)
- 降低匹配阈值

### 点击位置不准确
- 调整相应按钮的校准偏移值
- 修改默认位置比例

## 注意事项

1. 本脚本仅供学习和研究使用，请勿用于商业用途
2. 使用前请确保了解游戏的使用条款
3. 建议在测试环境中先验证脚本功能
4. 长时间运行可能会对游戏账号产生影响，请谨慎使用

## 开发计划

- [ ] 实现真正的OCR等级识别
- [ ] 添加更多游戏状态的检测
- [ ] 支持寰5
- [ ] 添加图形化配置界面
- [ ] 实现异常自动恢复功能

## 许可证

本项目采用MIT许可证，详情请参阅LICENSE文件。

## 贡献

欢迎提交Issue和Pull Request来改进这个项目。

## 支持


如果您在使用过程中遇到问题，请提交Issue或联系开发者。
