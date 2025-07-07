"""
字幕文本格式化处理器
用于在语音合成前对字幕文本进行预处理，使配音更自然
"""

import re
from typing import Dict, List, Tuple


class SubtitleTextFormatter:
    """字幕文本格式化处理器"""
    
    def __init__(self):
        # 数字连接符替换规则
        self.number_connectors = {
            '-': '至',
            '~': '到',
            '–': '至',  # en dash
            '—': '至',  # em dash
        }
        
        # 需要删除的括号内容模式
        self.bracket_patterns = [
            r'\([^)]*\)',      # 圆括号 (内容)
            r'\[[^\]]*\]',     # 方括号 [内容]
            r'\{[^}]*\}',      # 花括号 {内容}
            r'（[^）]*）',      # 中文圆括号 （内容）
            r'【[^】]*】',      # 中文方括号 【内容】
        ]
        
        # 标点符号优化
        self.punctuation_replacements = {
            '...': '。',
            '…': '。',
            '!!': '！',
            '??': '？',
            '；；': '；',
            '：：': '：',
        }
        
        # 英文缩写替换
        self.abbreviation_replacements = {
            'Mr.': '先生',
            'Mrs.': '女士',
            'Ms.': '女士',
            'Dr.': '博士',
            'Prof.': '教授',
            'etc.': '等等',
            'vs.': '对',
            'VS.': '对',
            'vs': '对',
            'VS': '对',
        }
        
        # 特殊符号替换
        self.symbol_replacements = {
            '&': '和',
            '@': '在',
            '#': '号',
            '%': '百分之',
            '$': '美元',
            '€': '欧元',
            '£': '英镑',
            '¥': '人民币',
            '°': '度',
            '±': '正负',
            '×': '乘以',
            '÷': '除以',
            '=': '等于',
            '+': '加',
            '∞': '无穷',
        }
    
    def format_text(self, text: str) -> str:
        """
        格式化字幕文本

        Args:
            text: 原始字幕文本

        Returns:
            格式化后的文本
        """
        if not text:
            return text

        # 如果只有空白字符，返回空字符串（让TTS系统处理）
        if not text.strip():
            return ""
        
        # 1. 移除多余的空白字符
        formatted_text = self._clean_whitespace(text)
        
        # 2. 处理数字连接符
        formatted_text = self._format_number_connectors(formatted_text)
        
        # 3. 删除括号内容
        formatted_text = self._remove_bracket_content(formatted_text)
        
        # 4. 替换英文缩写（在数字连接符之后，避免 vs. 被误处理）
        formatted_text = self._replace_abbreviations(formatted_text)

        # 5. 替换特殊符号
        formatted_text = self._replace_symbols(formatted_text)
        
        # 6. 优化标点符号
        formatted_text = self._optimize_punctuation(formatted_text)
        
        # 7. 最终清理
        formatted_text = self._final_cleanup(formatted_text)
        
        return formatted_text
    
    def _clean_whitespace(self, text: str) -> str:
        """清理多余的空白字符"""
        # 移除行首行尾空白
        text = text.strip()

        # 如果只有空白字符，直接返回空字符串
        if not text or text.isspace():
            return ""

        # 将多个连续空格替换为单个空格
        text = re.sub(r'\s+', ' ', text)

        return text
    
    def _format_number_connectors(self, text: str) -> str:
        """格式化数字连接符"""
        # 匹配数字-数字的模式
        for connector, replacement in self.number_connectors.items():
            # 匹配 数字+连接符+数字 的模式，但要考虑百分号等后缀
            pattern = rf'(\d+%?)\s*{re.escape(connector)}\s*(\d+%?)'
            text = re.sub(pattern, rf'\1{replacement}\2', text)

        return text
    
    def _remove_bracket_content(self, text: str) -> str:
        """删除括号内容，支持嵌套括号"""
        # 特殊处理嵌套圆括号
        while '(' in text and ')' in text:
            # 找到最内层的括号并删除
            text = re.sub(r'\([^()]*\)', '', text)

        # 处理其他类型的括号
        other_patterns = [
            r'\[[^\]]*\]',     # 方括号 [内容]
            r'\{[^}]*\}',      # 花括号 {内容}
            r'（[^）]*）',      # 中文圆括号 （内容）
            r'【[^】]*】',      # 中文方括号 【内容】
        ]

        for pattern in other_patterns:
            text = re.sub(pattern, '', text)

        return text
    
    def _replace_abbreviations(self, text: str) -> str:
        """替换英文缩写"""
        for abbr, replacement in self.abbreviation_replacements.items():
            # 直接替换，不使用单词边界（因为包含点号）
            text = text.replace(abbr, replacement)
            # 也处理大小写变体
            text = text.replace(abbr.upper(), replacement)
            text = text.replace(abbr.lower(), replacement)

        return text
    
    def _replace_symbols(self, text: str) -> str:
        """替换特殊符号"""
        for symbol, replacement in self.symbol_replacements.items():
            text = text.replace(symbol, replacement)
        
        return text
    
    def _optimize_punctuation(self, text: str) -> str:
        """优化标点符号"""
        for punct, replacement in self.punctuation_replacements.items():
            text = text.replace(punct, replacement)

        # 移除多余的标点符号，但保留至少一个
        text = re.sub(r'([。！？；：，])\1+', r'\1', text)

        return text
    
    def _final_cleanup(self, text: str) -> str:
        """最终清理"""
        # 移除多余的空格
        text = re.sub(r'\s+', ' ', text)

        # 移除标点符号前的空格
        text = re.sub(r'\s+([。！？；：，])', r'\1', text)

        # 移除行首行尾空白
        text = text.strip()

        # 如果文本为空或只有空白字符，返回空字符串（让TTS处理）
        if not text or text.isspace():
            return ""

        # 如果文本只有标点符号，返回空字符串（让TTS处理）
        if re.match(r'^[。！？；：，\s]*$', text):
            return ""

        return text
    
    def format_with_details(self, text: str) -> Dict[str, str]:
        """
        格式化文本并返回详细信息
        
        Args:
            text: 原始字幕文本
            
        Returns:
            包含原始文本、格式化文本和处理步骤的字典
        """
        original_text = text
        steps = []
        
        # 记录每个处理步骤
        current_text = text
        
        # 1. 清理空白字符
        new_text = self._clean_whitespace(current_text)
        if new_text != current_text:
            steps.append(f"清理空白: '{current_text}' → '{new_text}'")
            current_text = new_text
        
        # 2. 处理数字连接符
        new_text = self._format_number_connectors(current_text)
        if new_text != current_text:
            steps.append(f"数字连接符: '{current_text}' → '{new_text}'")
            current_text = new_text
        
        # 3. 删除括号内容
        new_text = self._remove_bracket_content(current_text)
        if new_text != current_text:
            steps.append(f"删除括号: '{current_text}' → '{new_text}'")
            current_text = new_text
        
        # 4. 替换缩写
        new_text = self._replace_abbreviations(current_text)
        if new_text != current_text:
            steps.append(f"替换缩写: '{current_text}' → '{new_text}'")
            current_text = new_text
        
        # 5. 替换符号
        new_text = self._replace_symbols(current_text)
        if new_text != current_text:
            steps.append(f"替换符号: '{current_text}' → '{new_text}'")
            current_text = new_text
        
        # 6. 优化标点
        new_text = self._optimize_punctuation(current_text)
        if new_text != current_text:
            steps.append(f"优化标点: '{current_text}' → '{new_text}'")
            current_text = new_text
        
        # 7. 最终清理
        new_text = self._final_cleanup(current_text)
        if new_text != current_text:
            steps.append(f"最终清理: '{current_text}' → '{new_text}'")
            current_text = new_text
        
        return {
            'original': original_text,
            'formatted': current_text,
            'steps': steps,
            'changed': original_text != current_text
        }


# 创建全局实例
subtitle_formatter = SubtitleTextFormatter()


def format_subtitle_text(text: str) -> str:
    """
    格式化字幕文本的便捷函数
    
    Args:
        text: 原始字幕文本
        
    Returns:
        格式化后的文本
    """
    return subtitle_formatter.format_text(text)


def format_subtitle_text_with_details(text: str) -> Dict[str, str]:
    """
    格式化字幕文本并返回详细信息的便捷函数
    
    Args:
        text: 原始字幕文本
        
    Returns:
        包含处理详情的字典
    """
    return subtitle_formatter.format_with_details(text)
