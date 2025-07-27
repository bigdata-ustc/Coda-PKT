from tree_sitter import Language, Parser

Language.build_library(
    'my-languages.so',
    [
        'tree-sitter-c',
        'tree-sitter-cpp',
        'tree-sitter-java',
        'tree-sitter-python',
    ]
)