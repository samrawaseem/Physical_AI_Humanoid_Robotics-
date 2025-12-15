// frontend/src/hooks/useTextSelection.ts
import { useState, useEffect, useCallback } from 'react';

interface TextSelection {
  text: string;
  startContainer: Node | null;
  endContainer: Node | null;
  startOffset: number;
  endOffset: number;
}

const useTextSelection = () => {
  const [selectedText, setSelectedText] = useState<string>('');
  const [selectionInfo, setSelectionInfo] = useState<TextSelection | null>(null);

  const getSelectedText = useCallback((): string => {
    const selection = window.getSelection();
    if (selection && selection.toString().trim() !== '') {
      return selection.toString().trim();
    }
    return '';
  }, []);

  const getSelectionInfo = useCallback((): TextSelection | null => {
    const selection = window.getSelection();
    if (selection && selection.rangeCount > 0 && selection.toString().trim() !== '') {
      const range = selection.getRangeAt(0);
      return {
        text: selection.toString().trim(),
        startContainer: range.startContainer,
        endContainer: range.endContainer,
        startOffset: range.startOffset,
        endOffset: range.endOffset,
      };
    }
    return null;
  }, []);

  const handleSelectionChange = useCallback(() => {
    const text = getSelectedText();
    setSelectedText(text);

    const info = getSelectionInfo();
    setSelectionInfo(info);
  }, [getSelectedText, getSelectionInfo]);

  useEffect(() => {
    document.addEventListener('selectionchange', handleSelectionChange);

    return () => {
      document.removeEventListener('selectionchange', handleSelectionChange);
    };
  }, [handleSelectionChange]);

  const clearSelection = useCallback(() => {
    const selection = window.getSelection();
    if (selection) {
      selection.removeAllRanges();
    }
    setSelectedText('');
    setSelectionInfo(null);
  }, []);

  return {
    selectedText,
    selectionInfo,
    getSelectedText,
    getSelectionInfo,
    clearSelection,
  };
};

export default useTextSelection;