<?php
define('DVWA_WEB_PAGE_TO_ROOT', '../../');
require_once DVWA_WEB_PAGE_TO_ROOT . 'dvwa/includes/dvwaPage.inc.php';
dvwaPageStartup(array('authenticated'));

$payload = (isset($_GET['payload']) && is_string($_GET['payload'])) ? $_GET['payload'] : '';
$ctx     = (isset($_GET['ctx']) && is_string($_GET['ctx'])) ? $_GET['ctx'] : 'html_body';
$family  = (isset($_GET['family']) && is_string($_GET['family'])) ? $_GET['family'] : '';
$carrier = (isset($_GET['carrier']) && is_string($_GET['carrier'])) ? $_GET['carrier'] : 'a';
$event   = (isset($_GET['event']) && is_string($_GET['event'])) ? $_GET['event'] : '';
$csp     = (isset($_GET['csp']) && is_string($_GET['csp'])) ? $_GET['csp'] : '0';
$enc     = (isset($_GET['enc']) && is_string($_GET['enc'])) ? $_GET['enc'] : '0';

header('Content-Type: text/html; charset=UTF-8');
header('X-XSS-Protection: 0');
if ($csp === '1') {
    header("Content-Security-Policy: script-src 'self'; object-src 'none';");
} else {
    header_remove('Content-Security-Policy');
}

$prefix = '<!doctype html><html><head><meta charset=utf-8>';
$style  = '<style>#oracle-host{min-height:80px} .oracle-focus{min-width:20px;min-height:20px}</style>';
$family_lower = strtolower($family);

if ($ctx === 'html_body') {
    if ($enc === '1') {
        $safe = htmlspecialchars($payload, ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8');
        $body = '<main id="oracle-host">' . $safe . '</main>';
    } else {
        if (in_array($event, array('onload', 'onpageshow'), true) && stripos($payload, '<body') !== false) {
            echo $prefix . $style . '</head>' . $payload . '</html>';
            exit;
        }
        $body = '<main id="oracle-host">' . $payload . '</main>';
    }
} elseif ($ctx === 'attribute') {
    $quote = (strpos($family_lower, 'attr_break_squote') === 0) ? "'" : '"';
    $val = ($enc === '1')
        ? htmlspecialchars($payload, ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8')
        : $payload;
    if (in_array($event, array('onkeyup', 'oninput', 'onchange', 'onfocus', 'onfocusin'), true)) {
        $body = '<input id="payload-target" type="text" class="oracle-focus" value=' . $quote . $val . $quote . '>';
    } else {
        $body = '<img id="payload-target" class="oracle-focus" tabindex="0" src="/__oracle_missing_resource__" value=' . $quote . $val . $quote . '>';
    }
} elseif ($ctx === 'js_string') {
    if ($enc === '1') {
        $json_flags = JSON_HEX_TAG | JSON_HEX_AMP | JSON_HEX_APOS | JSON_HEX_QUOT | JSON_UNESCAPED_UNICODE;
        if (defined('JSON_INVALID_UTF8_SUBSTITUTE')) {
            $json_flags |= JSON_INVALID_UTF8_SUBSTITUTE;
        }
        $json_safe = json_encode($payload, $json_flags);
        if ($json_safe === false) {
            $json_safe = '""';
        }
        if (strpos($family_lower, 'js_break_template') === 0) {
            $tpl_inner = substr($json_safe, 1, -1);
            $tpl_inner = str_replace(array('`', '$'), array('\\u0060', '\\u0024'), $tpl_inner);
            $script = 'var q = `' . $tpl_inner . '`;';
        } else {
            $script = 'var q = ' . $json_safe . ';';
        }
    } else {
        if (
            strpos($family_lower, 'js_direct_code_no_quote') === 0
            || strpos($family_lower, 'js_break_bracket_constructor') === 0
        ) {
            $script = $payload;
        } elseif (strpos($family_lower, 'js_break_comment') === 0) {
            $script = '/*' . $payload . '*/';
        } elseif (strpos($family_lower, 'js_break_template') === 0) {
            $script = 'var q = `' . $payload . '`;';
        } elseif (strpos($family_lower, 'js_break_dquote') === 0) {
            $script = 'var q = "' . $payload . '";';
        } elseif (strpos($family_lower, 'js_') === 0) {
            $p = $payload;
            if (strpos($family_lower, 'js_break_backslash_bypass') === 0) {
                $p = str_replace("'", "\\'", $p);
            }
            $script = "var q = '" . $p . "';";
        } else {
            $script = 'var q = "' . $payload . '";';
        }
    }
    $body = '<div id="payload-target" tabindex="0"></div><script>' . $script . '</script>';
} elseif ($ctx === 'url') {
    if ($enc === '1') {
        $candidate = trim($payload, "\x00..\x20\x7f");
        if ($candidate === '' || preg_match('/[\x00-\x1f\x7f]/', $candidate)) {
            $safe_url = '#';
        } elseif (preg_match('#^https?://#i', $candidate)) {
            $safe_url = $candidate;
        } elseif (
            $candidate[0] === '/'
            && (strlen($candidate) === 1 || ($candidate[1] !== '/' && $candidate[1] !== '\\'))
        ) {
            $safe_url = $candidate;
        } else {
            $safe_url = '#';
        }
        $href = htmlspecialchars($safe_url, ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8');
    } else {
        $href = $payload;
    }

    if ($carrier === 'iframe') {
        $body = '<iframe id="payload-target" src="' . $href . '"></iframe>';
    } elseif ($carrier === 'object') {
        $body = '<object id="payload-target" data="' . $href . '"></object>';
    } else {
        $body = '<a id="payload-target" href="' . $href . '">open payload</a>';
    }
} else {
    http_response_code(400);
    echo $prefix . $style . '</head><body><main id="oracle-host">Unsupported ctx</main></body></html>';
    exit;
}

echo $prefix . $style . '</head><body>' . $body . '</body></html>';
