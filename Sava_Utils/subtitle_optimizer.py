#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KlicStudio 字幕优化器 - 一体化版本
整合所有核心功能，提供便携的函数接口
"""

import re
import os
import unicodedata
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class SrtEntry:
    """SRT字幕条目"""
    index: int
    start_time: str
    end_time: str
    text: str


class SubtitleOptimizer:
    """字幕优化器 - 一体化核心处理器"""
    
    def __init__(self, max_sentence_length: int = 70, max_line_length: int = 35):
        """
        初始化字幕优化器
        
        Args:
            max_sentence_length: 最大句子长度
            max_line_length: 单行最大长度
        """
        self.max_sentence_length = max_sentence_length
        self.max_line_length = max_line_length
        
        # 成对标点符号映射 - 修复编译错误
        self.pair_punctuations = {
            '「': '」', 
            '『': '』', 
            '"': '"', 
            '\'': '\'',
            '《': '》', 
            '<': '>', 
            '【': '】', 
            '〔': '〕',
            '(': ')', 
            '[': ']', 
            '{': '}',
        }
        
        # 需要处理的单标点
        self.single_punctuations = ",.;:!?~，、。！？；：…"
        
        # 亚洲语言代码
        self.asian_languages = {'zh_cn', 'zh_tw', 'ja', 'ko', 'th'}
    
    def split_text_sentences(self, text: str) -> List[str]:
        """智能文本切分"""
        if not text.strip():
            return []
        
        # 占位符
        DOT_PLACEHOLDER = "\u0001"
        COMMA_PLACEHOLDER = "\u0002"
        TIME_PLACEHOLDER = "\u0003"
        
        # 时间格式保护 (如 3:30 p.m.)
        time_pattern = re.compile(r'\b\d{1,2}(?::|\.)\d{2}\s+[ap]\.m\.', re.IGNORECASE)
        text = time_pattern.sub(lambda m: m.group().replace('.', TIME_PLACEHOLDER), text)
        
        # 千位分隔符保护 (如 1,000)
        thousand_pattern = re.compile(r'\b\d{1,3}(?:,\d{3})+\b')
        text = thousand_pattern.sub(lambda m: m.group().replace(',', COMMA_PLACEHOLDER), text)
        
        # 小数保护 (如 3.14)
        decimal_pattern = re.compile(r'\b\d+\.\d+\b')
        text = decimal_pattern.sub(lambda m: m.group().replace('.', DOT_PLACEHOLDER), text)
        
        # 缩写词保护 (如 U.S.A.)
        abbrev_pattern = re.compile(r'\b(?:[A-Za-z]\.){2,}[A-Za-z]?\b|\b[A-Z][a-z]*\.(?:[A-Z][a-z]*\.)+')
        text = abbrev_pattern.sub(lambda m: m.group().replace('.', DOT_PLACEHOLDER), text)
        
        # 按标点符号切分
        split_pattern = re.compile(r'([。.！!？?；;，,\n]+)')
        text = split_pattern.sub(r'\1\u0000', text)
        parts = text.split('\u0000')
        
        sentences = []
        for part in parts:
            s = part.strip()
            # 恢复被保护的字符
            s = s.replace(TIME_PLACEHOLDER, '.')
            s = s.replace(DOT_PLACEHOLDER, '.')
            s = s.replace(COMMA_PLACEHOLDER, ',')
            if s:
                sentences.append(s)
        
        return sentences
    
    def calc_length(self, text: str) -> float:
        """计算文本视觉长度 - 支持多语言字符权重"""
        length = 0.0
        for char in text:
            code = ord(char)
            if 0x4E00 <= code <= 0x9FFF or 0x3040 <= code <= 0x30FF:  # 中日文
                length += 1.75
            elif 0xAC00 <= code <= 0xD7A3 or 0x1100 <= code <= 0x11FF:  # 韩文
                length += 1.5
            elif 0x0E00 <= code <= 0x0E7F:  # 泰文
                length += 1.0
            elif 0xFF01 <= code <= 0xFF5E:  # 全角符号
                length += 1.75
            else:  # 其他字符（英文等）
                length += 1.0
        return length
    
    def beautify_asian_language_sentence(self, text: str) -> str:
        """亚洲语言符号美化处理"""
        if not text:
            return text
        
        # 处理字符串末尾的标点
        chars = list(text)
        i = len(chars) - 1
        
        while i >= 0:
            char = chars[i]
            if char.isspace():
                i -= 1
                continue
            if char in self.single_punctuations:
                chars.pop(i)
                i -= 1
            else:
                break
        
        # 中间的单标点替换为空格
        in_pair = False
        expected_close = None
        result = []
        
        for char in chars:
            # 检查是否是成对标点的开始
            if char in self.pair_punctuations and not in_pair:
                in_pair = True
                expected_close = self.pair_punctuations[char]
                result.append(char)
                continue
            
            # 检查是否是成对标点的结束
            if in_pair and char == expected_close:
                in_pair = False
                expected_close = None
                result.append(char)
                continue
            
            # 在成对标点内部，保持原样
            if in_pair:
                result.append(char)
                continue
            
            # 不在成对标点内，处理单标点
            if char in self.single_punctuations:
                # 替换为空格，但避免连续空格
                if result and result[-1] != ' ':
                    result.append(' ')
            else:
                result.append(char)
        
        return ''.join(result).strip()
    
    def split_text_by_length(self, text: str, language: str = 'en') -> List[str]:
        """根据长度智能切分文本"""
        if self.calc_length(text) <= self.max_sentence_length:
            return [text]
        
        # 基于标点符号的切分
        sentences = self.split_text_sentences(text)
        if len(sentences) > 1:
            return sentences
        
        # 如果没有标点符号，按长度强制切分
        if self.is_asian_language(language):
            return self._split_asian_text_by_length(text)
        else:
            return self._split_western_text_by_length(text)
    
    def _split_asian_text_by_length(self, text: str) -> List[str]:
        """按长度切分亚洲语言文本"""
        result = []
        current = ""
        
        for char in text:
            if self.calc_length(current + char) <= self.max_sentence_length:
                current += char
            else:
                if current:
                    result.append(current.strip())
                current = char
        
        if current:
            result.append(current.strip())
        
        return [s for s in result if s]
    
    def _split_western_text_by_length(self, text: str) -> List[str]:
        """按长度切分西方语言文本"""
        words = text.split()
        result = []
        current = ""
        
        for word in words:
            test_text = f"{current} {word}".strip()
            if self.calc_length(test_text) <= self.max_sentence_length:
                current = test_text
            else:
                if current:
                    result.append(current)
                current = word
        
        if current:
            result.append(current)
        
        return result
    
    def is_asian_language(self, code: str) -> bool:
        """判断是否为亚洲语言"""
        return code in self.asian_languages
    
    def parse_srt_file(self, file_path: str) -> List[SrtEntry]:
        """解析SRT字幕文件"""
        entries = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='gbk') as f:
                    content = f.read()
            except UnicodeDecodeError:
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()
        
        # 分割字幕块
        blocks = re.split(r'\n\s*\n', content.strip())
        
        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) >= 3:
                try:
                    index = int(lines[0])
                    time_line = lines[1]
                    text = '\n'.join(lines[2:])
                    
                    # 解析时间戳
                    time_match = re.match(r'(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})', time_line)
                    if time_match:
                        start_time = time_match.group(1)
                        end_time = time_match.group(2)
                        
                        entries.append(SrtEntry(
                            index=index,
                            start_time=start_time,
                            end_time=end_time,
                            text=text
                        ))
                except (ValueError, IndexError):
                    continue
        
        return entries
    
    def write_srt_file(self, entries: List[SrtEntry], file_path: str):
        """写入SRT字幕文件"""
        with open(file_path, 'w', encoding='utf-8') as f:
            for entry in entries:
                f.write(f"{entry.index}\n")
                f.write(f"{entry.start_time} --> {entry.end_time}\n")
                f.write(f"{entry.text}\n\n")
    
    def optimize_subtitle_entry(self, entry: SrtEntry, language: str = 'auto') -> List[SrtEntry]:
        """优化单个字幕条目"""
        text = entry.text
        
        # 自动检测语言
        if language == 'auto':
            language = self._detect_language(text)
        
        # 应用亚洲语言美化
        if self.is_asian_language(language):
            text = self.beautify_asian_language_sentence(text)
        
        # 检查长度并切分
        if self.calc_length(text) > self.max_line_length:
            split_texts = self.split_text_by_length(text, language)
            
            # 为每个切分的文本创建新的条目
            result = []
            for i, split_text in enumerate(split_texts):
                new_entry = SrtEntry(
                    index=entry.index + i * 0.01,
                    start_time=entry.start_time,
                    end_time=entry.end_time,
                    text=split_text
                )
                result.append(new_entry)
            return result
        else:
            return [SrtEntry(
                index=entry.index,
                start_time=entry.start_time,
                end_time=entry.end_time,
                text=text
            )]
    
    def _detect_language(self, text: str) -> str:
        """简单的语言检测"""
        # 统计不同语言字符的比例
        total_chars = len(text)
        if total_chars == 0:
            return 'en'
        
        chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
        japanese_chars = sum(1 for char in text if '\u3040' <= char <= '\u309f' or '\u30a0' <= char <= '\u30ff')
        korean_chars = sum(1 for char in text if '\uac00' <= char <= '\ud7a3')
        
        if chinese_chars / total_chars > 0.3:
            return 'zh_cn'
        elif japanese_chars / total_chars > 0.2:
            return 'ja'
        elif korean_chars / total_chars > 0.2:
            return 'ko'
        else:
            return 'en'


# ==================== 便携函数接口 ====================

def optimize_srt_file(input_path: str, output_path: str = None,
                     max_sentence_length: int = 70, max_line_length: int = 35,
                     language: str = 'auto') -> str:
    """
    优化SRT字幕文件 - 便携函数接口

    Args:
        input_path: 输入SRT文件路径
        output_path: 输出SRT文件路径，如果为None则自动生成
        max_sentence_length: 最大句子长度
        max_line_length: 单行最大长度
        language: 语言代码，'auto'为自动检测

    Returns:
        输出文件路径
    """
    # 自动生成输出路径
    if output_path is None:
        base_name = os.path.splitext(input_path)[0]
        output_path = f"{base_name}_optimized.srt"

    # 创建优化器
    optimizer = SubtitleOptimizer(max_sentence_length, max_line_length)

    # 解析输入文件
    entries = optimizer.parse_srt_file(input_path)

    # 优化所有条目
    optimized_entries = []
    for entry in entries:
        optimized = optimizer.optimize_subtitle_entry(entry, language)
        optimized_entries.extend(optimized)

    # 重新编号
    for i, entry in enumerate(optimized_entries, 1):
        entry.index = i

    # 写入输出文件
    optimizer.write_srt_file(optimized_entries, output_path)

    return output_path


def beautify_asian_subtitles(input_path: str, output_path: str = None) -> str:
    """
    美化亚洲语言字幕符号 - 便携函数

    Args:
        input_path: 输入SRT文件路径
        output_path: 输出SRT文件路径

    Returns:
        输出文件路径
    """
    if output_path is None:
        base_name = os.path.splitext(input_path)[0]
        output_path = f"{base_name}_beautified.srt"

    optimizer = SubtitleOptimizer()
    entries = optimizer.parse_srt_file(input_path)

    # 只进行符号美化
    for entry in entries:
        language = optimizer._detect_language(entry.text)
        if optimizer.is_asian_language(language):
            entry.text = optimizer.beautify_asian_language_sentence(entry.text)

    optimizer.write_srt_file(entries, output_path)
    return output_path


def split_long_subtitles(input_path: str, output_path: str = None,
                        max_length: int = 35) -> str:
    """
    切分过长字幕 - 便携函数

    Args:
        input_path: 输入SRT文件路径
        output_path: 输出SRT文件路径
        max_length: 最大长度

    Returns:
        输出文件路径
    """
    if output_path is None:
        base_name = os.path.splitext(input_path)[0]
        output_path = f"{base_name}_split.srt"

    optimizer = SubtitleOptimizer(max_line_length=max_length)
    entries = optimizer.parse_srt_file(input_path)

    # 切分长字幕
    split_entries = []
    for entry in entries:
        if optimizer.calc_length(entry.text) > max_length:
            language = optimizer._detect_language(entry.text)
            split_texts = optimizer.split_text_by_length(entry.text, language)

            for i, text in enumerate(split_texts):
                new_entry = SrtEntry(
                    index=entry.index + i * 0.01,
                    start_time=entry.start_time,
                    end_time=entry.end_time,
                    text=text
                )
                split_entries.append(new_entry)
        else:
            split_entries.append(entry)

    # 重新编号
    for i, entry in enumerate(split_entries, 1):
        entry.index = i

    optimizer.write_srt_file(split_entries, output_path)
    return output_path


def get_subtitle_stats(input_path: str) -> Dict[str, Any]:
    """
    获取字幕统计信息 - 便携函数

    Args:
        input_path: SRT文件路径

    Returns:
        统计信息字典
    """
    optimizer = SubtitleOptimizer()
    entries = optimizer.parse_srt_file(input_path)

    if not entries:
        return {}

    total_chars = sum(len(entry.text) for entry in entries)
    total_visual_length = sum(optimizer.calc_length(entry.text) for entry in entries)

    # 计算时长
    def time_to_seconds(time_str):
        pattern = r'(\d{2}):(\d{2}):(\d{2}),(\d{3})'
        match = re.match(pattern, time_str)
        if match:
            hours, minutes, seconds, milliseconds = map(int, match.groups())
            return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000
        return 0.0

    total_duration = (time_to_seconds(entries[-1].end_time) -
                     time_to_seconds(entries[0].start_time))

    # 统计长字幕
    long_subtitles = [e for e in entries if optimizer.calc_length(e.text) > optimizer.max_line_length]

    # 语言检测
    languages = {}
    for entry in entries:
        lang = optimizer._detect_language(entry.text)
        languages[lang] = languages.get(lang, 0) + 1

    return {
        'total_entries': len(entries),
        'total_characters': total_chars,
        'total_visual_length': total_visual_length,
        'total_duration_seconds': total_duration,
        'average_chars_per_entry': total_chars / len(entries),
        'average_visual_length_per_entry': total_visual_length / len(entries),
        'long_subtitles_count': len(long_subtitles),
        'long_subtitles_percentage': len(long_subtitles) / len(entries) * 100,
        'detected_languages': languages,
        'chars_per_second': total_chars / total_duration if total_duration > 0 else 0
    }


def batch_optimize_directory(input_dir: str, output_dir: str = None, **kwargs):
    """
    批量优化目录中的SRT文件 - 便携函数

    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        **kwargs: 优化参数
    """
    if output_dir is None:
        output_dir = os.path.join(input_dir, 'optimized')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    processed_count = 0
    error_count = 0

    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.srt'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"optimized_{filename}")

            try:
                optimize_srt_file(input_path, output_path, **kwargs)
                print(f"✓ 处理完成: {filename}")
                processed_count += 1
            except Exception as e:
                print(f"✗ 处理失败 {filename}: {e}")
                error_count += 1

    print(f"\n批量处理完成: 成功 {processed_count} 个，失败 {error_count} 个")


# ==================== 使用示例 ====================

def demo():
    """演示函数使用"""
    print("=== KlicStudio 字幕优化器演示 ===\n")

    # 创建示例SRT文件
    sample_content = """1
00:00:01,000 --> 00:00:04,000
Hello everyone, welcome to our presentation today. This is going to be a very interesting and informative session that will cover multiple topics.

2
00:00:04,500 --> 00:00:07,000
你好大家，欢迎来到我们今天的演示。。。这将是一个非常有趣和信息丰富的会议！！！

3
00:00:07,500 --> 00:00:09,000
Let's get started!

4
00:00:09,500 --> 00:00:12,000
让我们开始吧！
"""

    # 创建示例文件
    with open('demo_input.srt', 'w', encoding='utf-8') as f:
        f.write(sample_content)

    print("1. 创建了示例文件: demo_input.srt")

    # 获取统计信息
    stats = get_subtitle_stats('demo_input.srt')
    print(f"\n2. 字幕统计信息:")
    print(f"   总条目数: {stats['total_entries']}")
    print(f"   总字符数: {stats['total_characters']}")
    print(f"   平均视觉长度: {stats['average_visual_length_per_entry']:.1f}")
    print(f"   长字幕数量: {stats['long_subtitles_count']}")
    print(f"   检测到的语言: {stats['detected_languages']}")

    # 优化字幕
    output_file = optimize_srt_file('demo_input.srt')
    print(f"\n3. 优化完成，输出文件: {output_file}")

    # 美化亚洲语言字幕
    beautified_file = beautify_asian_subtitles('demo_input.srt')
    print(f"4. 符号美化完成，输出文件: {beautified_file}")

    # 切分长字幕
    split_file = split_long_subtitles('demo_input.srt', max_length=30)
    print(f"5. 长字幕切分完成，输出文件: {split_file}")

    print("\n=== 演示完成 ===")


if __name__ == "__main__":
    demo()
