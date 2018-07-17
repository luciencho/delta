# coding:utf-8
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

stop_word_list = [
    "--", "?", "“", "”", "》", "－－", "able", "about",
    "above", "according", "accordingly", "across", "actually", "after", "afterwards", "again",
    "against", "ain't", "all", "allow", "allows", "almost", "alone", "along",
    "already", "also", "although", "always", "am", "among", "amongst", "an",
    "and", "another", "any", "anybody", "anyhow", "anyone", "anything", "anyway",
    "anyways", "anywhere", "apart", "appear", "appreciate", "appropriate", "are", "aren't",
    "around", "as", "a's", "aside", "ask", "asking", "associated", "at",
    "available", "away", "awfully", "be", "became", "because", "become", "becomes",
    "becoming", "been", "before", "beforehand", "behind", "being", "believe", "below",
    "beside", "besides", "best", "better", "between", "beyond", "both", "brief",
    "but", "by", "came", "can", "cannot", "cant", "can't", "cause",
    "causes", "certain", "certainly", "changes", "clearly", "c'mon", "co", "com",
    "come", "comes", "concerning", "consequently", "consider", "considering", "contain", "containing",
    "contains", "corresponding", "could", "couldn't", "course", "c's", "currently", "definitely",
    "described", "despite", "did", "didn't", "different", "do", "does", "doesn't",
    "doing", "done", "don't", "down", "downwards", "during", "each", "edu",
    "eg", "eight", "either", "else", "elsewhere", "enough", "entirely", "especially",
    "et", "etc", "even", "ever", "every", "everybody", "everyone", "everything",
    "everywhere", "ex", "exactly", "example", "except", "far", "few", "fifth",
    "first", "five", "followed", "following", "follows", "for", "former", "formerly",
    "forth", "four", "from", "further", "furthermore", "get", "gets", "getting",
    "given", "gives", "go", "goes", "going", "gone", "got", "gotten",
    "greetings", "had", "hadn't", "happens", "hardly", "has", "hasn't", "have",
    "haven't", "having", "he", "hello", "help", "hence", "her", "here",
    "hereafter", "hereby", "herein", "here's", "hereupon", "hers", "herself", "he's",
    "hi", "him", "himself", "his", "hither", "hopefully", "how", "howbeit",
    "however", "i'd", "ie", "if", "ignored", "i'll", "i'm", "immediate",
    "in", "inasmuch", "inc", "indeed", "indicate", "indicated", "indicates", "inner",
    "insofar", "instead", "into", "inward", "is", "isn't", "it", "it'd",
    "it'll", "its", "it's", "itself", "i've", "just", "keep", "keeps",
    "kept", "know", "known", "knows", "last", "lately", "later", "latter",
    "latterly", "least", "less", "lest", "let", "let's", "like", "liked",
    "likely", "little", "look", "looking", "looks", "ltd", "mainly", "many",
    "may", "maybe", "me", "mean", "meanwhile", "merely", "might", "more",
    "moreover", "most", "mostly", "much", "must", "my", "myself", "name",
    "namely", "nd", "near", "nearly", "necessary", "need", "needs", "neither",
    "never", "nevertheless", "new", "next", "nine", "no", "nobody", "non",
    "none", "noone", "nor", "normally", "not", "nothing", "novel", "now",
    "nowhere", "obviously", "of", "off", "often", "oh", "ok", "okay",
    "old", "on", "once", "one", "ones", "only", "onto", "or",
    "other", "others", "otherwise", "ought", "our", "ours", "ourselves", "out",
    "outside", "over", "overall", "own", "particular", "particularly", "per", "perhaps",
    "placed", "please", "plus", "possible", "presumably", "probably", "provides", "que",
    "quite", "qv", "rather", "rd", "re", "really", "reasonably", "regarding",
    "regardless", "regards", "relatively", "respectively", "right", "said", "same", "saw",
    "say", "saying", "says", "second", "secondly", "see", "seeing", "seem",
    "seemed", "seeming", "seems", "seen", "self", "selves", "sensible", "sent",
    "serious", "seriously", "seven", "several", "shall", "she", "should", "shouldn't",
    "since", "six", "so", "some", "somebody", "somehow", "someone", "something",
    "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "specified", "specify",
    "specifying", "still", "sub", "such", "sup", "sure", "take", "taken",
    "tell", "tends", "th", "than", "thank", "thanks", "thanx", "that",
    "thats", "that's", "the", "their", "theirs", "them", "themselves", "then",
    "thence", "there", "thereafter", "thereby", "therefore", "therein", "theres", "there's",
    "thereupon", "these", "they", "they'd", "they'll", "they're", "they've", "think",
    "third", "this", "thorough", "thoroughly", "those", "though", "three", "through",
    "throughout", "thru", "thus", "to", "together", "too", "took", "toward",
    "towards", "tried", "tries", "truly", "try", "trying", "t's", "twice",
    "two", "un", "under", "unfortunately", "unless", "unlikely", "until", "unto",
    "up", "upon", "us", "use", "used", "useful", "uses", "using",
    "usually", "value", "various", "very", "via", "viz", "vs", "want",
    "wants", "was", "wasn't", "way", "we", "we'd", "welcome", "well",
    "we'll", "went", "were", "we're", "weren't", "we've", "what", "whatever",
    "what's", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby",
    "wherein", "where's", "whereupon", "wherever", "whether", "which", "while", "whither",
    "who", "whoever", "whole", "whom", "who's", "whose", "why", "will",
    "willing", "wish", "with", "within", "without", "wonder", "won't", "would",
    "wouldn't", "yes", "yet", "you", "you'd", "you'll", "your", "you're",
    "yours", "yourself", "yourselves", "you've", "zero", "zt", "ZT", "zz",
    "ZZ", "一", "一下", "一些", "一切", "一则", "一天", "一定",
    "一方面", "一旦", "一时", "一来", "一样", "一次", "一片", "一直",
    "一致", "一般", "一起", "一边", "一面", "万一", "上下", "上升",
    "上去", "上来", "上述", "上面", "下列", "下去", "下来", "下面",
    "不一", "不久", "不仅", "不会", "不但", "不光", "不单", "不变",
    "不只", "不可", "不同", "不够", "不如", "不得", "不怕", "不惟",
    "不成", "不拘", "不敢", "不断", "不是", "不比", "不然", "不特",
    "不独", "不管", "不能", "不要", "不论", "不足", "不过", "不问",
    "与", "与其", "与否", "与此同时", "专门", "且", "两者", "严格",
    "严重", "个", "个人", "个别", "中小", "中间", "丰富", "临",
    "为", "为主", "为了", "为什么", "为什麽", "为何", "为着", "主张",
    "主要", "举行", "乃", "乃至", "么", "之", "之一", "之前",
    "之后", "之後", "之所以", "之类", "乌乎", "乎", "乘", "也",
    "也好", "也是", "也罢", "了", "了解", "争取", "于", "于是",
    "于是乎", "云云", "互相", "产生", "人们", "人家", "什么", "什么样",
    "什麽", "今后", "今天", "今年", "今後", "仍然", "从", "从事",
    "从而", "他", "他人", "他们", "他的", "代替", "以", "以上",
    "以下", "以为", "以便", "以免", "以前", "以及", "以后", "以外",
    "以後", "以来", "以至", "以至于", "以致", "们", "任", "任何",
    "任凭", "任务", "企图", "伟大", "似乎", "似的", "但", "但是",
    "何", "何况", "何处", "何时", "作为", "你", "你们", "你的",
    "使得", "使用", "例如", "依", "依照", "依靠", "促进", "保持",
    "俺", "俺们", "倘", "倘使", "倘或", "倘然", "倘若", "假使",
    "假如", "假若", "做到", "像", "允许", "充分", "先后", "先後",
    "先生", "全部", "全面", "兮", "共同", "关于", "其", "其一",
    "其中", "其二", "其他", "其余", "其它", "其实", "其次", "具体",
    "具体地说", "具体说来", "具有", "再者", "再说", "冒", "冲", "决定",
    "况且", "准备", "几", "几乎", "几时", "凭", "凭借", "出去",
    "出来", "出现", "分别", "则", "别", "别的", "别说", "到",
    "前后", "前者", "前进", "前面", "加之", "加以", "加入", "加强",
    "十分", "即", "即令", "即使", "即便", "即或", "即若", "却不",
    "原来", "又", "及", "及其", "及时", "及至", "双方", "反之",
    "反应", "反映", "反过来", "反过来说", "取得", "受到", "变成", "另",
    "另一方面", "另外", "只是", "只有", "只要", "只限", "叫", "叫做",
    "召开", "叮咚", "可", "可以", "可是", "可能", "可见", "各",
    "各个", "各人", "各位", "各地", "各种", "各级", "各自", "合理",
    "同", "同一", "同时", "同样", "后来", "后面", "向", "向着",
    "吓", "吗", "否则", "吧", "吧哒", "吱", "呀", "呃",
    "呕", "呗", "呜", "呜呼", "呢", "周围", "呵", "呸",
    "呼哧", "咋", "和", "咚", "咦", "咱", "咱们", "咳",
    "哇", "哈", "哈哈", "哉", "哎", "哎呀", "哎哟", "哗",
    "哟", "哦", "哩", "哪", "哪个", "哪些", "哪儿", "哪天",
    "哪年", "哪怕", "哪样", "哪边", "哪里", "哼", "哼唷", "唉",
    "啊", "啐", "啥", "啦", "啪达", "喂", "喏", "喔唷",
    "嗡嗡", "嗬", "嗯", "嗳", "嘎", "嘎登", "嘘", "嘛",
    "嘻", "嘿", "因", "因为", "因此", "因而", "固然", "在",
    "在下", "地", "坚决", "坚持", "基本", "处理", "复杂", "多",
    "多少", "多数", "多次", "大力", "大多数", "大大", "大家", "大批",
    "大约", "大量", "失去", "她", "她们", "她的", "好的", "好象",
    "如", "如上所述", "如下", "如何", "如其", "如果", "如此", "如若",
    "存在", "宁", "宁可", "宁愿", "宁肯", "它", "它们", "它们的",
    "它的", "安全", "完全", "完成", "实现", "实际", "宣布", "容易",
    "密切", "对", "对于", "对应", "将", "少数", "尔后", "尚且",
    "尤其", "就", "就是", "就是说", "尽", "尽管", "属于", "岂但",
    "左右", "巨大", "巩固", "己", "已经", "帮助", "常常", "并",
    "并不", "并不是", "并且", "并没有", "广大", "广泛", "应当", "应用",
    "应该", "开外", "开始", "开展", "引起", "强烈", "强调", "归",
    "当", "当前", "当时", "当然", "当着", "形成", "彻底", "彼",
    "彼此", "往", "往往", "待", "後来", "後面", "得", "得出",
    "得到", "心里", "必然", "必要", "必须", "怎", "怎么", "怎么办",
    "怎么样", "怎样", "怎麽", "总之", "总是", "总的来看", "总的来说", "总的说来",
    "总结", "总而言之", "恰恰相反", "您", "意思", "愿意", "慢说", "成为",
    "我", "我们", "我的", "或", "或是", "或者", "战斗", "所",
    "所以", "所有", "所谓", "打", "扩大", "把", "抑或", "拿",
    "按", "按照", "换句话说", "换言之", "据", "掌握", "接着", "接著",
    "故", "故此", "整个", "方便", "方面", "旁人", "无宁", "无法",
    "无论", "既", "既是", "既然", "时候", "明显", "明确", "是",
    "是否", "是的", "显然", "显著", "普通", "普遍", "更加", "曾经",
    "替", "最后", "最大", "最好", "最後", "最近", "最高", "有",
    "有些", "有关", "有利", "有力", "有所", "有效", "有时", "有点",
    "有的", "有着", "有著", "望", "朝", "朝着", "本", "本着",
    "来", "来着", "极了", "构成", "果然", "果真", "某", "某个",
    "某些", "根据", "根本", "欢迎", "正在", "正如", "正常", "此",
    "此外", "此时", "此间", "毋宁", "每", "每个", "每天", "每年",
    "每当", "比", "比如", "比方", "比较", "毫不", "没有", "沿",
    "沿着", "注意", "深入", "清楚", "满足", "漫说", "焉", "然则",
    "然后", "然後", "然而", "照", "照着", "特别是", "特殊", "特点",
    "现代", "现在", "甚么", "甚而", "甚至", "用", "由", "由于",
    "由此可见", "的", "的话", "目前", "直到", "直接", "相似", "相信",
    "相反", "相同", "相对", "相对而言", "相应", "相当", "相等", "省得",
    "看出", "看到", "看来", "看看", "看见", "真是", "真正", "着",
    "着呢", "矣", "知道", "确定", "离", "积极", "移动", "突出",
    "突然", "立即", "第", "等", "等等", "管", "紧接着", "纵",
    "纵令", "纵使", "纵然", "练习", "组成", "经", "经常", "经过",
    "结合", "结果", "给", "绝对", "继续", "继而", "维持", "综上所述",
    "罢了", "考虑", "者", "而", "而且", "而况", "而外", "而已",
    "而是", "而言", "联系", "能", "能否", "能够", "腾", "自",
    "自个儿", "自从", "自各儿", "自家", "自己", "自身", "至", "至于",
    "良好", "若", "若是", "若非", "范围", "莫若", "获得", "虽",
    "虽则", "虽然", "虽说", "行为", "行动", "表明", "表示", "被",
    "要", "要不", "要不是", "要不然", "要么", "要是", "要求", "规定",
    "觉得", "认为", "认真", "认识", "让", "许多", "论", "设使",
    "设若", "该", "说明", "诸位", "谁", "谁知", "赶", "起",
    "起来", "起见", "趁", "趁着", "越是", "跟", "转动", "转变",
    "转贴", "较", "较之", "边", "达到", "迅速", "过", "过去",
    "过来", "运用", "还是", "还有", "这", "这个", "这么", "这么些",
    "这么样", "这么点儿", "这些", "这会儿", "这儿", "这就是说", "这时", "这样",
    "这点", "这种", "这边", "这里", "这麽", "进入", "进步", "进而",
    "进行", "连", "连同", "适应", "适当", "适用", "逐步", "逐渐",
    "通常", "通过", "造成", "遇到", "遭到", "避免", "那", "那个",
    "那么", "那么些", "那么样", "那些", "那会儿", "那儿", "那时", "那样",
    "那边", "那里", "那麽", "部分", "鄙人", "采取", "里面", "重大",
    "重新", "重要", "鉴于", "问题", "防止", "阿", "附近", "限制",
    "除", "除了", "除此之外", "除非", "随", "随着", "随著", "集中",
    "需要", "非但", "非常", "非徒", "靠", "顺", "顺着", "首先",
    "高兴", "是不是", "说说", "———", "》），", "）÷（１－", "”，", "）、",
    "＝（", ":", "→", "℃ ", "&", "*", "一一", "~~~~",
    "’", ". ", "『", ".一", "./", "-- ", "』", "＝″",
    "【", "［＊］", "｝＞", "［⑤］］", "［①Ｄ］", "ｃ］", "ｎｇ昉", "＊",
    "//", "［", "］", "［②ｅ］", "［②ｇ］", "＝｛", "}", "，也 ",
    "‘", "Ａ", "［①⑥］", "［②Ｂ］ ", "［①ａ］", "［④ａ］", "［①③］", "［③ｈ］",
    "③］", "１． ", "－－ ", "［②ｂ］", "’‘ ", "××× ", "［①⑧］", "０：２ ",
    "＝［", "［⑤ｂ］", "［②ｃ］ ", "［④ｂ］", "［②③］", "［③ａ］", "［④ｃ］", "［①⑤］",
    "［①⑦］", "［①ｇ］", "∈［ ", "［①⑨］", "［①④］", "［①ｃ］", "［②ｆ］", "［②⑧］",
    "［②①］", "［①Ｃ］", "［③ｃ］", "［③ｇ］", "［②⑤］", "［②②］", "一.", "［①ｈ］",
    ".数", "［］", "［①Ｂ］", "数/", "［①ｉ］", "［③ｅ］", "［①①］", "［④ｄ］",
    "［④ｅ］", "［③ｂ］", "［⑤ａ］", "［①Ａ］", "［②⑧］", "［②⑦］", "［①ｄ］", "［②ｊ］",
    "〕〔", "］［", "://", "′∈", "［②④", "［⑤ｅ］", "１２％", "ｂ］",
    "...", "...................", "…………………………………………………③", "ＺＸＦＩＴＬ", "［③Ｆ］", "」", "［①ｏ］", "］∧′＝［ ",
    "∪φ∈", "′｜", "｛－", "②ｃ", "｝", "［③①］", "Ｒ．Ｌ．", "［①Ｅ］",
    "Ψ", "－［＊］－", "↑", ".日 ", "［②ｄ］", "［②", "［②⑦］", "［②②］",
    "［③ｅ］", "［①ｉ］", "［①Ｂ］", "［①ｈ］", "［①ｄ］", "［①ｇ］", "［①②］", "［②ａ］",
    "ｆ］", "［⑩］", "ａ］", "［①ｅ］", "［②ｈ］", "［②⑥］", "［③ｄ］", "［②⑩］",
    "ｅ］", "〉", "】", "元／吨", "［②⑩］", "２．３％", "５：０  ", "［①］",
    "::", "［②］", "［③］", "［④］", "［⑤］", "［⑥］", "［⑦］", "［⑧］",
    "［⑨］ ", "……", "——", "?", "、", "。", "“", "”",
    "《", "》", "！", "，", "：", "；", "？", "．",
    ",", "．", "'", "? ", "·", "———", "──", "? ",
    "—", "<", ">", "（", "）", "〔", "〕", "[",
    "]", "(", ")", "-", "+", "～", "×", "／",
    "/", "①", "②", "③", "④", "⑤", "⑥", "⑦",
    "⑧", "⑨", "⑩", "Ⅲ", "В", '"', ";", "#",
    "@", "γ", "μ", "φ", "φ．", "× ", "Δ", "■",
    "▲", "sub", "exp ", "sup", "sub", "Lex ", "＃", "％",
    "＆", "＇", "＋", "＋ξ", "＋＋", "－", "－β", "＜",
    "＜±", "＜Δ", "＜λ", "＜φ", "＜＜", "=", "＝", "＝☆",
    "＝－", "＞", "＞λ", "＿", "～±", "～＋", "［⑤ｆ］", "［⑤ｄ］",
    "［②ｉ］", "≈ ", "［②Ｇ］", "［①ｆ］", "ＬＩ", "㈧ ", "［－", "......",
    "〉", "［③⑩］", "第二", "一番", "一直", "一个", "一些", "许多",
    "种", "有的是", "也就是说", "末##末", "啊", "阿", "哎", "哎呀",
    "哎哟", "唉", "俺", "俺们", "按", "按照", "吧", "吧哒",
    "把", "罢了", "被", "本", "本着", "比", "比方", "比如",
    "鄙人", "彼", "彼此", "边", "别", "别的", "别说", "并",
    "并且", "不比", "不成", "不单", "不但", "不独", "不管", "不光",
    "不过", "不仅", "不拘", "不论", "不怕", "不然", "不如", "不特",
    "不惟", "不问", "不只", "朝", "朝着", "趁", "趁着", "乘",
    "冲", "除", "除此之外", "除非", "除了", "此", "此间", "此外",
    "从", "从而", "打", "待", "但", "但是", "当", "当着",
    "到", "得", "的", "的话", "等", "等等", "地", "第",
    "叮咚", "对", "对于", "多", "多少", "而", "而况", "而且",
    "而是", "而外", "而言", "而已", "尔后", "反过来", "反过来说", "反之",
    "非但", "非徒", "否则", "嘎", "嘎登", "该", "赶", "个",
    "各", "各个", "各位", "各种", "各自", "给", "根据", "跟",
    "故", "故此", "固然", "关于", "管", "归", "果然", "果真",
    "过", "哈", "哈哈", "呵", "和", "何", "何处", "何况",
    "何时", "嘿", "哼", "哼唷", "呼哧", "乎", "哗", "还是",
    "还有", "换句话说", "换言之", "或", "或是", "或者", "极了", "及",
    "及其", "及至", "即", "即便", "即或", "即令", "即若", "即使",
    "几", "几时", "己", "既", "既然", "既是", "继而", "加之",
    "假如", "假若", "假使", "鉴于", "将", "较", "较之", "叫",
    "接着", "结果", "借", "紧接着", "进而", "尽", "尽管", "经",
    "经过", "就", "就是", "就是说", "据", "具体地说", "具体说来", "开始",
    "开外", "靠", "咳", "可", "可见", "可是", "可以", "况且",
    "啦", "来", "来着", "离", "例如", "哩", "连", "连同",
    "两者", "了", "临", "另", "另外", "另一方面", "论", "嘛",
    "吗", "慢说", "漫说", "冒", "么", "每", "每当", "们",
    "莫若", "某", "某个", "某些", "拿", "哪", "哪边", "哪儿",
    "哪个", "哪里", "哪年", "哪怕", "哪天", "哪些", "哪样", "那",
    "那边", "那儿", "那个", "那会儿", "那里", "那么", "那么些", "那么样",
    "那时", "那些", "那样", "乃", "乃至", "呢", "能", "你",
    "你们", "您", "宁", "宁可", "宁肯", "宁愿", "哦", "呕",
    "啪达", "旁人", "呸", "凭", "凭借", "其", "其次", "其二",
    "其他", "其它", "其一", "其余", "其中", "起", "起见", "起见",
    "岂但", "恰恰相反", "前后", "前者", "且", "然而", "然后", "然则",
    "让", "人家", "任", "任何", "任凭", "如", "如此", "如果",
    "如何", "如其", "如若", "如上所述", "若", "若非", "若是", "啥",
    "上下", "尚且", "设若", "设使", "甚而", "甚么", "甚至", "省得",
    "时候", "什么", "什么样", "使得", "是", "是的", "首先", "谁",
    "谁知", "顺", "顺着", "似的", "虽", "虽然", "虽说", "虽则",
    "随", "随着", "所", "所以", "他", "他们", "他人", "它",
    "它们", "她", "她们", "倘", "倘或", "倘然", "倘若", "倘使",
    "腾", "替", "通过", "同", "同时", "哇", "万一", "往",
    "望", "为", "为何", "为了", "为什么", "为着", "喂", "嗡嗡",
    "我", "我们", "呜", "呜呼", "乌乎", "无论", "无宁", "毋宁",
    "嘻", "吓", "相对而言", "像", "向", "向着", "嘘", "呀",
    "焉", "沿", "沿着", "要", "要不", "要不然", "要不是", "要么",
    "要是", "也", "也罢", "也好", "一", "一般", "一旦", "一方面",
    "一来", "一切", "一样", "一则", "依", "依照", "矣", "以",
    "以便", "以及", "以免", "以至", "以至于", "以致", "抑或", "因",
    "因此", "因而", "因为", "哟", "用", "由", "由此可见", "由于",
    "有", "有的", "有关", "有些", "又", "于", "于是", "于是乎",
    "与", "与此同时", "与否", "与其", "越是", "云云", "哉", "再说",
    "再者", "在", "在下", "咱", "咱们", "则", "怎", "怎么",
    "怎么办", "怎么样", "怎样", "咋", "照", "照着", "者", "这",
    "这边", "这儿", "这个", "这会儿", "这就是说", "这里", "这么", "这么点儿",
    "这么些", "这么样", "这时", "这些", "这样", "正如", "吱", "之",
    "之类", "之所以", "之一", "只是", "只限", "只要", "只有", "至",
    "至于", "诸位", "着", "着呢", "自", "自从", "自个儿", "自各儿",
    "自己", "自家", "自身", "综上所述", "总的来看", "总的来说", "总的说来", "总而言之",
    "总之", "纵", "纵令", "纵然", "纵使", "遵照", "作为", "兮",
    "呃", "呗", "咚", "咦", "喏", "啐", "喔唷", "嗬",
    "嗯", "嗳", "打开天窗说亮话", "到目前为止", "赶早不赶晚", "常言说得好", "何乐而不为", "毫无保留地",
    "由此可见", "这就是说", "这么点儿", "综上所述", "总的来看", "总的来说", "总的说来", "总而言之",
    "相对而言", "除此之外", "反过来说", "恰恰相反", "如上所述", "换句话说", "具体地说", "具体说来",
    "另一方面", "与此同时", "一则通过", "毫无例外", "不然的话", "从此以后", "从古到今", "从古至今",
    "从今以后", "大张旗鼓", "从无到有", "从早到晚", "弹指之间", "不亦乐乎", "不知不觉", "不止一次",
    "不择手段", "不可开交", "不可抗拒", "不仅仅是", "不管怎样", "挨家挨户", "长此下去", "长话短说",
    "除此而外", "除此以外", "除此之外", "得天独厚", "川流不息", "长期以来", "挨门挨户", "挨门逐户",
    "多多少少", "多多益善", "二话不说", "更进一步", "二话没说", "分期分批", "风雨无阻", "归根到底",
    "归根结底", "反之亦然", "大面儿上", "倒不如说", "成年累月", "换句话说", "或多或少", "简而言之",
    "接连不断", "尽如人意", "尽心竭力", "尽心尽力", "尽管如此", "据我所知", "具体地说", "具体来说",
    "具体说来", "近几年来", "每时每刻", "屡次三番", "三番两次", "三番五次", "三天两头", "另一方面",
    "老老实实", "年复一年", "恰恰相反", "顷刻之间", "穷年累月", "千万千万", "日复一日", "如此等等",
    "如前所述", "如上所述", "一方面", "切不可", "顷刻间", "全身心", "另方面", "另一个",
    "猛然间", "默默地", "就是说", "近年来", "尽可能", "接下来", "简言之", "急匆匆",
    "即是说", "基本上", "换言之", "充其极", "充其量", "暗地里", "反之则", "比如说",
    "背地里", "背靠背", "并没有", "不得不", "不得了", "不得已", "不仅仅", "不经意",
    "不能不", "不外乎", "不由得", "不怎么", "不至于", "策略地", "差不多", "常言道",
    "常言说", "多年来", "多年前", "差一点", "敞开儿", "抽冷子", "大不了", "反倒是",
    "反过来", "大体上", "当口儿", "倒不如", "怪不得", "动不动", "看起来", "看上去",
    "看样子", "够瞧的", "到了儿", "呆呆地", "来不及", "来得及", "到头来", "连日来",
    "于是乎", "为什么", "这会儿", "换言之", "那会儿", "那么些", "那么样", "什么样",
    "反过来", "紧接着", "就是说", "要不然", "要不是", "一方面", "以至于", "自个儿",
    "自各儿", "之所以", "这么些", "这么样", "怎么办", "怎么样", "谁知", "顺着",
    "似的", "虽然", "虽说", "虽则", "随着", "所以", "他们", "他人",
    "它们", "她们", "倘或", "倘然", "倘若", "倘使", "要么", "要是",
    "也罢", "也好", "以便", "依照", "以及", "以免", "以至", "以致",
    "抑或", "因此", "因而", "因为", "由于", "有的", "有关", "有些",
    "于是", "与否", "与其", "越是", "云云", "一般", "一旦", "一来",
    "一切", "一样", "同时", "万一", "为何", "为了", "为着", "嗡嗡",
    "我们", "呜呼", "乌乎", "无论", "无宁", "沿着", "毋宁", "向着",
    "照着", "怎么", "咱们", "在下", "再说", "再者", "怎样", "这边",
    "这儿", "这个", "这里", "这么", "这时", "这些", "这样", "正如",
    "之类", "之一", "只是", "只限", "只要", "只有", "至于", "诸位",
    "着呢", "纵令", "纵然", "纵使", "遵照", "作为", "喔唷", "自从",
    "自己", "自家", "自身", "总之", "要不", "哎呀", "哎哟", "俺们",
    "按照", "吧哒", "罢了", "本着", "比方", "比如", "鄙人", "彼此",
    "别的", "别说", "并且", "不比", "不成", "不单", "不但", "不独",
    "不管", "不光", "不过", "不仅", "不拘", "不论", "不怕", "不然",
    "不如", "不特", "不惟", "不问", "不只", "朝着", "趁着", "除非",
    "除了", "此间", "此外", "从而", "但是", "当着", "的话", "等等",
    "叮咚", "对于", "多少", "而况", "而且", "而是", "而外", "而言",
    "而已", "尔后", "反之", "非但", "非徒", "否则", "嘎登", "各个",
    "各位", "各种", "各自", "根据", "故此", "固然", "关于", "果然",
    "果真", "哈哈", "何处", "何况", "何时", "哼唷", "呼哧", "还是",
    "还有", "或是", "或者", "极了", "及其", "及至", "即便", "即或",
    "即令", "即若", "即使", "既然", "既是", "继而", "加之", "假如",
    "假若", "假使", "鉴于", "几时", "较之", "接着", "结果", "进而",
    "尽管", "经过", "就是", "可见", "可是", "可以", "况且", "开始",
    "开外", "来着", "例如", "连同", "两者", "另外", "慢说", "漫说",
    "每当", "莫若", "某个", "某些", "哪边", "哪儿", "哪个", "哪里",
    "哪年", "哪怕", "哪天", "哪些", "哪样", "那边", "那儿", "那个",
    "那里", "那么", "那时", "那些", "那样", "乃至", "宁可", "宁肯",
    "宁愿", "你们", "啪达", "旁人", "凭借", "其次", "其二", "其他",
    "其它", "其一", "其余", "其中", "起见", "起见", "岂但", "前后",
    "前者", "然而", "然后", "然则", "人家", "任何", "任凭", "如此",
    "如果", "如何", "如其", "如若", "若非", "若是", "上下", "尚且",
    "设若", "设使", "甚而", "甚么", "甚至", "省得", "时候", "什么",
    "使得", "是的", "首先", "首先", "其次", "再次", "最后", "您们",
    "它们", "她们", "他们", "我们", "你是", "您是", "我是", "他是",
    "她是", "它是", "不是", "你们", "啊哈", "啊呀", "啊哟", "挨次",
    "挨个", "挨着", "哎呀", "哎哟", "俺们", "按理", "按期", "默然",
    "按时", "按说", "按照", "暗中", "暗自", "昂然", "八成", "倍感",
    "倍加", "本人", "本身", "本着", "并非", "别人", "必定", "比起",
    "比如", "比照", "鄙人", "毕竟", "必将", "必须", "并肩", "并没",
    "并排", "并且", "并无", "勃然", "不必", "不常", "不大", "不单",
    "不但", "而且", "不得", "不迭", "不定", "不独", "不对", "不妨",
    "不管", "不光", "不过", "不会", "不仅", "不拘", "不力", "不了",
    "不料", "不论", "不满", "不免", "不起", "不巧", "不然", "不日",
    "不少", "不胜", "不时", "不是", "不同", "不能", "不要", "不外",
    "不下", "不限", "不消", "不已", "不再", "不曾", "不止", "不只",
    "才能", "彻夜", "趁便", "趁机", "趁热", "趁势", "趁早", "趁着",
    "成心", "乘机", "乘势", "乘隙", "乘虚", "诚然", "迟早", "充分",
    "出来", "出去", "除此", "除非", "除开", "除了", "除去", "除却",
    "除外", "处处", "传说", "传闻", "纯粹", "此后", "此间", "此外",
    "此中", "次第", "匆匆", "从不", "从此", "从而", "从宽", "从来",
    "从轻", "从速", "从头", "从未", "从小", "从新", "从严", "从优",
    "从中", "从重", "凑巧", "存心", "达旦", "打从", "大大", "大抵",
    "大都", "大多", "大凡", "大概", "大家", "大举", "大略", "大约",
    "大致", "待到", "单纯", "单单", "但是", "但愿", "当场", "当儿",
    "当即", "当然", "当庭", "当头", "当下", "当真", "当中", "当着",
    "倒是", "到处", "到底", "到头", "得起", "的话", "的确", "等到",
    "等等", "顶多", "动辄", "陡然", "独自", "断然", "对于", "顿时",
    "多次", "多多", "多亏", "而后", "而论", "而且", "而是", "而外",
    "而言", "而已", "而又", "尔等", "反倒", "反而", "反手", "反之",
    "方才", "方能", "非常", "非但", "非得", "分头", "奋勇", "愤然",
    "更为", "更加", "根据", "个人", "各式", "刚才", "敢情", "该当",
    "嘎嘎", "否则", "赶快", "敢于", "刚好", "刚巧", "高低", "格外",
    "隔日", "隔夜", "公然", "过于", "果然", "果真", "光是", "关于",
    "共总", "姑且", "故此", "故而", "故意", "固然", "惯常", "毫不",
    "毫无", "很多", "何须", "好在", "何必", "何尝", "何妨", "何苦",
    "何况", "何止", "很少", "轰然", "后来", "呼啦", "哗啦", "互相",
    "忽地", "忽然", "话说", "或是", "伙同", "豁然", "恍然", "还是",
    "或许", "或者", "基本", "基于", "极大", "极度", "极端", "极力",
    "极其", "极为", "即便", "即将", "及其", "及至", "即刻", "即令",
    "即使", "几度", "几番", "几乎", "几经", "既然", "继而", "继之",
    "加上", "加以", "加之", "假如", "假若", "假使", "间或", "将才",
    "简直", "鉴于", "将近", "将要", "交口", "较比", "较为", "较之",
    "皆可", "截然", "截至", "藉以", "借此", "借以", "届时", "尽快",
    "近来", "进而", "进来", "进去", "尽管", "尽量", "尽然", "就算",
    "居然", "就此", "就地", "竟然", "究竟", "经常", "尽早", "精光",
    "经过", "就是", "局外", "举凡", "据称", "据此", "据实", "据说",
    "可好", "看来", "开外", "绝不", "决不", "据悉", "决非", "绝顶",
    "绝对", "绝非", "可见", "可能", "可是", "可以", "恐怕", "来讲",
    "来看", "快要", "况且", "拦腰", "牢牢", "老是", "累次", "累年",
    "理当", "理该", "理应", "例如", "立地", "立刻", "立马", "立时",
    "联袂", "连连", "连日", "路经", "临到", "连声", "连同", "连袂",
    "另外", "另行", "屡次", "屡屡", "缕缕", "率尔", "率然", "略加",
    "略微", "略为", "论说", "马上", "猛然", "没有", "每当", "每逢",
    "每每", "莫不", "莫非", "莫如", "莫若", "哪怕", "那么", "那末",
    "那些", "乃至", "难道", "难得", "难怪", "难说", "你们", "凝神",
    "宁可", "宁肯", "宁愿", "偶而", "偶尔", "碰巧", "譬如", "偏偏",
    "平素", "迫于", "扑通", "其次", "其后", "其实", "其它", "起初",
    "起来", "起首", "起头", "起先", "岂但", "岂非", "岂止", "恰逢",
    "恰好", "恰恰", "恰巧", "恰如", "恰似", "前后", "前者", "切莫",
    "切切", "切勿", "亲口", "亲身", "亲手", "亲眼", "亲自", "顷刻",
    "请勿", "取道", "权时", "全都", "全力", "全年", "全然", "然而",
    "然后", "人家", "人人", "仍旧", "仍然", "日见", "日渐", "日益",
    "日臻", "如常", "如次", "如果", "如今", "如期", "如若", "如上",
    "如下", "上来", "上去", "瑟瑟", "沙沙", "啊", "哎", "唉",
    "俺", "按", "吧", "把", "甭", "别", "嘿", "很",
    "乎", "会", "或", "既", "及", "啦", "了", "们",
    "你", "您", "哦", "砰", "啊", "你", "我", "他",
    "她", "它", "$", "0", "1", "2", "3", "4",
    "5", "6", "7", "8", "9", "?", "_", "“",
    "”", "、", "。", "《", "》", "一", "一些", "一何",
    "一切", "一则", "一方面", "一旦", "一来", "一样", "一般", "一转眼",
    "万一", "上", "上下", "下", "不", "不仅", "不但", "不光",
    "不单", "不只", "不外乎", "不如", "不妨", "不尽", "不尽然", "不得",
    "不怕", "不惟", "不成", "不拘", "不料", "不是", "不比", "不然",
    "不特", "不独", "不管", "不至于", "不若", "不论", "不过", "不问",
    "与", "与其", "与其说", "与否", "与此同时", "且", "且不说", "且说",
    "两者", "个", "个别", "临", "为", "为了", "为什么", "为何",
    "为止", "为此", "为着", "乃", "乃至", "乃至于", "么", "之",
    "之一", "之所以", "之类", "乌乎", "乎", "乘", "也", "也好",
    "也罢", "了", "二来", "于", "于是", "于是乎", "云云", "云尔",
    "些", "亦", "人", "人们", "人家", "什么", "什么样", "今",
    "介于", "仍", "仍旧", "从", "从此", "从而", "他", "他人",
    "他们", "以", "以上", "以为", "以便", "以免", "以及", "以故",
    "以期", "以来", "以至", "以至于", "以致", "们", "任", "任何",
    "任凭", "似的", "但", "但凡", "但是", "何", "何以", "何况",
    "何处", "何时", "余外", "作为", "你", "你们", "使", "使得",
    "例如", "依", "依据", "依照", "便于", "俺", "俺们", "倘",
    "倘使", "倘或", "倘然", "倘若", "借", "假使", "假如", "假若",
    "傥然", "像", "儿", "先不先", "光是", "全体", "全部", "兮",
    "关于", "其", "其一", "其中", "其二", "其他", "其余", "其它",
    "其次", "具体地说", "具体说来", "兼之", "内", "再", "再其次", "再则",
    "再有", "再者", "再者说", "再说", "冒", "冲", "况且", "几",
    "几时", "凡", "凡是", "凭", "凭借", "出于", "出来", "分别",
    "则", "则甚", "别", "别人", "别处", "别是", "别的", "别管",
    "别说", "到", "前后", "前此", "前者", "加之", "加以", "即",
    "即令", "即使", "即便", "即如", "即或", "即若", "却", "去",
    "又", "又及", "及", "及其", "及至", "反之", "反而", "反过来",
    "反过来说", "受到", "另", "另一方面", "另外", "另悉", "只", "只当",
    "只怕", "只是", "只有", "只消", "只要", "只限", "叫", "叮咚",
    "可", "可以", "可是", "可见", "各", "各个", "各位", "各种",
    "各自", "同", "同时", "后", "后者", "向", "向使", "向着",
    "吓", "吗", "否则", "吧", "吧哒", "吱", "呀", "呃",
    "呕", "呗", "呜", "呜呼", "呢", "呵", "呵呵", "呸",
    "呼哧", "咋", "和", "咚", "咦", "咧", "咱", "咱们",
    "咳", "哇", "哈", "哈哈", "哉", "哎", "哎呀", "哎哟",
    "哗", "哟", "哦", "哩", "哪", "哪个", "哪些", "哪儿",
    "哪天", "哪年", "哪怕", "哪样", "哪边", "哪里", "哼", "哼唷",
    "唉", "唯有", "啊", "啐", "啥", "啦", "啪达", "啷当",
    "喂", "喏", "喔唷", "喽", "嗡", "嗡嗡", "嗬", "嗯",
    "嗳", "嘎", "嘎登", "嘘", "嘛", "嘻", "嘿", "嘿嘿",
    "因", "因为", "因了", "因此", "因着", "因而", "固然", "在",
    "在下", "在于", "地", "基于", "处在", "多", "多么", "多少",
    "大", "大家", "她", "她们", "好", "如", "如上", "如上所述",
    "如下", "如何", "如其", "如同", "如是", "如果", "如此", "如若",
    "始而", "孰料", "孰知", "宁", "宁可", "宁愿", "宁肯", "它",
    "它们", "对", "对于", "对待", "对方", "对比", "将", "小",
    "尔", "尔后", "尔尔", "尚且", "就", "就是", "就是了", "就是说",
    "就算", "就要", "尽", "尽管", "尽管如此", "岂但", "己", "已",
    "已矣", "巴", "巴巴", "并", "并且", "并非", "庶乎", "庶几",
    "开外", "开始", "归", "归齐", "当", "当地", "当然", "当着",
    "彼", "彼时", "彼此", "往", "待", "很", "得", "得了",
    "怎", "怎么", "怎么办", "怎么样", "怎奈", "怎样", "总之", "总的来看",
    "总的来说", "总的说来", "总而言之", "恰恰相反", "您", "惟其", "慢说", "我",
    "我们", "或", "或则", "或是", "或曰", "或者", "截至", "所",
    "所以", "所在", "所幸", "所有", "才", "才能", "打", "打从",
    "把", "抑或", "拿", "按", "按照", "换句话说", "换言之", "据",
    "据此", "接着", "故", "故此", "故而", "旁人", "无", "无宁",
    "无论", "既", "既往", "既是", "既然", "时候", "是", "是以",
    "是的", "曾", "替", "替代", "最", "有", "有些", "有关",
    "有及", "有时", "有的", "望", "朝", "朝着", "本", "本人",
    "本地", "本着", "本身", "来", "来着", "来自", "来说", "极了",
    "果然", "果真", "某", "某个", "某些", "某某", "根据", "欤",
    "正值", "正如", "正巧", "正是", "此", "此地", "此处", "此外",
    "此时", "此次", "此间", "毋宁", "每", "每当", "比", "比及",
    "比如", "比方", "没奈何", "沿", "沿着", "漫说", "焉", "然则",
    "然后", "然而", "照", "照着", "犹且", "犹自", "甚且", "甚么",
    "甚或", "甚而", "甚至", "甚至于", "用", "用来", "由", "由于",
    "由是", "由此", "由此可见", "的", "的确", "的话", "直到", "相对而言",
    "省得", "看", "眨眼", "着", "着呢", "矣", "矣乎", "矣哉",
    "离", "竟而", "第", "等", "等到", "等等", "简言之", "管",
    "类如", "紧接着", "纵", "纵令", "纵使", "纵然", "经", "经过",
    "结果", "给", "继之", "继后", "继而", "综上所述", "罢了", "者",
    "而", "而且", "而况", "而后", "而外", "而已", "而是", "而言",
    "能", "能否", "腾", "自", "自个儿", "自从", "自各儿", "自后",
    "自家", "自己", "自打", "自身", "至", "至于", "至今", "至若",
    "致", "般的", "若", "若夫", "若是", "若果 ", "若非", "莫不然",
    "莫如", "莫若", "虽", "虽则", "虽然", "虽说", "被", "要",
    "要不", "要不是", "要不然", "要么", "要是", "譬喻", "譬如", "让",
    "许多", "论", "设使", "设或", "设若", "诚如", "诚然", "该",
    "说来", "诸", "诸位", "诸如", "谁", "谁人", "谁料", "谁知",
    "贼死", "赖以", "赶", "起", "起见", "趁", "趁着", "越是",
    "距", "跟", "较", "较之", "边", "过", "还", "还是",
    "还有", "还要", "这", "这一来", "这个", "这么", "这么些", "这么样",
    "这么点儿", "这些", "这会儿", "这儿", "这就是说", "这时", "这样", "这次",
    "这般", "这边", "这里", "进而", "连", "连同", "逐步", "通过",
    "遵循", "遵照", "那", "那个", "那么", "那么些", "那么样", "那些",
    "那会儿", "那儿", "那时", "那样", "那般", "那边", "那里", "都",
    "鄙人", "鉴于", "针对", "阿", "除", "除了", "除外", "除开",
    "除此之外", "除非", "随", "随后", "随时", "随着", "难道说", "非但",
    "非徒", "非特", "非独", "靠", "顺", "顺着", "首先", "！"
]
