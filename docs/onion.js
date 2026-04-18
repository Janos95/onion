/* @ts-self-types="./onion.d.ts" */
function __wbg_get_imports() {
    const import0 = {
        __proto__: null,
        __wbg___wbindgen_boolean_get_6ea149f0a8dcc5ff: function(arg0) {
            const v = arg0;
            const ret = typeof(v) === 'boolean' ? v : undefined;
            return isLikeNone(ret) ? 0xFFFFFF : ret ? 1 : 0;
        },
        __wbg___wbindgen_debug_string_ab4b34d23d6778bd: function(arg0, arg1) {
            const ret = debugString(arg1);
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg___wbindgen_is_undefined_29a43b4d42920abd: function(arg0) {
            const ret = arg0 === undefined;
            return ret;
        },
        __wbg___wbindgen_memory_dfa12096f400c9bd: function() {
            const ret = wasm.memory;
            return ret;
        },
        __wbg___wbindgen_number_get_c7f42aed0525c451: function(arg0, arg1) {
            const obj = arg1;
            const ret = typeof(obj) === 'number' ? obj : undefined;
            getDataViewMemory0().setFloat64(arg0 + 8 * 1, isLikeNone(ret) ? 0 : ret, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, !isLikeNone(ret), true);
        },
        __wbg___wbindgen_string_get_7ed5322991caaec5: function(arg0, arg1) {
            const obj = arg1;
            const ret = typeof(obj) === 'string' ? obj : undefined;
            var ptr1 = isLikeNone(ret) ? 0 : passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            var len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg___wbindgen_throw_6b64449b9b9ed33c: function(arg0, arg1) {
            throw new Error(getStringFromWasm0(arg0, arg1));
        },
        __wbg__wbg_cb_unref_b46c9b5a9f08ec37: function(arg0) {
            arg0._wbg_cb_unref();
        },
        __wbg_activeTexture_7b5b783dbad88ee6: function(arg0, arg1) {
            arg0.activeTexture(arg1 >>> 0);
        },
        __wbg_activeTexture_925d3f120338ace5: function(arg0, arg1) {
            arg0.activeTexture(arg1 >>> 0);
        },
        __wbg_addEventListener_861dfdaebdc6c705: function() { return handleError(function (arg0, arg1, arg2, arg3, arg4) {
            arg0.addEventListener(getStringFromWasm0(arg1, arg2), arg3, arg4);
        }, arguments); },
        __wbg_addEventListener_a08f4a083f97b81a: function() { return handleError(function (arg0, arg1, arg2, arg3) {
            arg0.addEventListener(getStringFromWasm0(arg1, arg2), arg3);
        }, arguments); },
        __wbg_addListener_b08b89b7cfd209de: function() { return handleError(function (arg0, arg1) {
            arg0.addListener(arg1);
        }, arguments); },
        __wbg_altKey_0b19b89ebfda349d: function(arg0) {
            const ret = arg0.altKey;
            return ret;
        },
        __wbg_altKey_9e14faa37fc6073c: function(arg0) {
            const ret = arg0.altKey;
            return ret;
        },
        __wbg_appendChild_f33cb3cfa3572859: function() { return handleError(function (arg0, arg1) {
            const ret = arg0.appendChild(arg1);
            return ret;
        }, arguments); },
        __wbg_attachShader_836a4cda0403b06c: function(arg0, arg1, arg2) {
            arg0.attachShader(arg1, arg2);
        },
        __wbg_attachShader_8dc540de24a1f185: function(arg0, arg1, arg2) {
            arg0.attachShader(arg1, arg2);
        },
        __wbg_beginQuery_6eae0bfd74320884: function(arg0, arg1, arg2) {
            arg0.beginQuery(arg1 >>> 0, arg2);
        },
        __wbg_bindBufferRange_2b0ac754d5560380: function(arg0, arg1, arg2, arg3, arg4, arg5) {
            arg0.bindBufferRange(arg1 >>> 0, arg2 >>> 0, arg3, arg4, arg5);
        },
        __wbg_bindBuffer_51e77d566f87252b: function(arg0, arg1, arg2) {
            arg0.bindBuffer(arg1 >>> 0, arg2);
        },
        __wbg_bindBuffer_5ab4eb61fe9bc86d: function(arg0, arg1, arg2) {
            arg0.bindBuffer(arg1 >>> 0, arg2);
        },
        __wbg_bindFramebuffer_a393c21fa07609d9: function(arg0, arg1, arg2) {
            arg0.bindFramebuffer(arg1 >>> 0, arg2);
        },
        __wbg_bindFramebuffer_f19b46d05ec6c1bb: function(arg0, arg1, arg2) {
            arg0.bindFramebuffer(arg1 >>> 0, arg2);
        },
        __wbg_bindRenderbuffer_804ce15356334930: function(arg0, arg1, arg2) {
            arg0.bindRenderbuffer(arg1 >>> 0, arg2);
        },
        __wbg_bindRenderbuffer_89719db4ed63ae22: function(arg0, arg1, arg2) {
            arg0.bindRenderbuffer(arg1 >>> 0, arg2);
        },
        __wbg_bindSampler_c6d50d2247164bf4: function(arg0, arg1, arg2) {
            arg0.bindSampler(arg1 >>> 0, arg2);
        },
        __wbg_bindTexture_1aaa433d602438c8: function(arg0, arg1, arg2) {
            arg0.bindTexture(arg1 >>> 0, arg2);
        },
        __wbg_bindTexture_f77bde66fec86f78: function(arg0, arg1, arg2) {
            arg0.bindTexture(arg1 >>> 0, arg2);
        },
        __wbg_bindVertexArrayOES_2210353006c0aff8: function(arg0, arg1) {
            arg0.bindVertexArrayOES(arg1);
        },
        __wbg_bindVertexArray_2aefe50c26a4ae47: function(arg0, arg1) {
            arg0.bindVertexArray(arg1);
        },
        __wbg_blendColor_505fb2cb2604165e: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.blendColor(arg1, arg2, arg3, arg4);
        },
        __wbg_blendColor_b7ccb8ee98011210: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.blendColor(arg1, arg2, arg3, arg4);
        },
        __wbg_blendEquationSeparate_22de603e88465350: function(arg0, arg1, arg2) {
            arg0.blendEquationSeparate(arg1 >>> 0, arg2 >>> 0);
        },
        __wbg_blendEquationSeparate_b259d64eaab74fbd: function(arg0, arg1, arg2) {
            arg0.blendEquationSeparate(arg1 >>> 0, arg2 >>> 0);
        },
        __wbg_blendEquation_3d3d3db4dc58ffb7: function(arg0, arg1) {
            arg0.blendEquation(arg1 >>> 0);
        },
        __wbg_blendEquation_b208a68703b0751b: function(arg0, arg1) {
            arg0.blendEquation(arg1 >>> 0);
        },
        __wbg_blendFuncSeparate_7e6270715f874cf0: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.blendFuncSeparate(arg1 >>> 0, arg2 >>> 0, arg3 >>> 0, arg4 >>> 0);
        },
        __wbg_blendFuncSeparate_d5250dad713c766c: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.blendFuncSeparate(arg1 >>> 0, arg2 >>> 0, arg3 >>> 0, arg4 >>> 0);
        },
        __wbg_blendFunc_357465c4dd812859: function(arg0, arg1, arg2) {
            arg0.blendFunc(arg1 >>> 0, arg2 >>> 0);
        },
        __wbg_blendFunc_976ec83c228404cb: function(arg0, arg1, arg2) {
            arg0.blendFunc(arg1 >>> 0, arg2 >>> 0);
        },
        __wbg_blitFramebuffer_5c63fb995f98768b: function(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10) {
            arg0.blitFramebuffer(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9 >>> 0, arg10 >>> 0);
        },
        __wbg_body_6026b8f025bfcb84: function(arg0) {
            const ret = arg0.body;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_bufferData_98b898f8977b1009: function(arg0, arg1, arg2, arg3) {
            arg0.bufferData(arg1 >>> 0, arg2, arg3 >>> 0);
        },
        __wbg_bufferData_9cf8a4b3e2b112b6: function(arg0, arg1, arg2, arg3) {
            arg0.bufferData(arg1 >>> 0, arg2, arg3 >>> 0);
        },
        __wbg_bufferData_beb80da5aad2402c: function(arg0, arg1, arg2, arg3) {
            arg0.bufferData(arg1 >>> 0, arg2, arg3 >>> 0);
        },
        __wbg_bufferData_f34b1b7feb55a62a: function(arg0, arg1, arg2, arg3) {
            arg0.bufferData(arg1 >>> 0, arg2, arg3 >>> 0);
        },
        __wbg_bufferSubData_1ac41f22281114ee: function(arg0, arg1, arg2, arg3) {
            arg0.bufferSubData(arg1 >>> 0, arg2, arg3);
        },
        __wbg_bufferSubData_5e737d8e869ad3d1: function(arg0, arg1, arg2, arg3) {
            arg0.bufferSubData(arg1 >>> 0, arg2, arg3);
        },
        __wbg_buffer_7447b9cc2267a9e5: function(arg0) {
            const ret = arg0.buffer;
            return ret;
        },
        __wbg_button_6d1e6f557d99f0d2: function(arg0) {
            const ret = arg0.button;
            return ret;
        },
        __wbg_buttons_ff15118d6e9f58ba: function(arg0) {
            const ret = arg0.buttons;
            return ret;
        },
        __wbg_call_b51415974987aa44: function() { return handleError(function (arg0, arg1) {
            const ret = arg0.call(arg1);
            return ret;
        }, arguments); },
        __wbg_cancelAnimationFrame_04b88e6554b7d10b: function() { return handleError(function (arg0, arg1) {
            arg0.cancelAnimationFrame(arg1);
        }, arguments); },
        __wbg_cancelBubble_9e225114ed0ead3b: function(arg0) {
            const ret = arg0.cancelBubble;
            return ret;
        },
        __wbg_charCode_13923a19ddab3037: function(arg0) {
            const ret = arg0.charCode;
            return ret;
        },
        __wbg_clearBufferfi_a5c3fb00067efabc: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.clearBufferfi(arg1 >>> 0, arg2, arg3, arg4);
        },
        __wbg_clearBufferfv_bd416dde8777ef2a: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.clearBufferfv(arg1 >>> 0, arg2, getArrayF32FromWasm0(arg3, arg4));
        },
        __wbg_clearBufferiv_e8dcff72b3a318ea: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.clearBufferiv(arg1 >>> 0, arg2, getArrayI32FromWasm0(arg3, arg4));
        },
        __wbg_clearBufferuiv_be7a8a3fe3171c61: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.clearBufferuiv(arg1 >>> 0, arg2, getArrayU32FromWasm0(arg3, arg4));
        },
        __wbg_clearTimeout_5902f3b9d0d5880c: function(arg0, arg1) {
            arg0.clearTimeout(arg1);
        },
        __wbg_clientWaitSync_c2dcb63719991bf8: function(arg0, arg1, arg2, arg3) {
            const ret = arg0.clientWaitSync(arg1, arg2 >>> 0, arg3 >>> 0);
            return ret;
        },
        __wbg_clientX_7ce2870ced210810: function(arg0) {
            const ret = arg0.clientX;
            return ret;
        },
        __wbg_clientY_cf4837c432ecd2ae: function(arg0) {
            const ret = arg0.clientY;
            return ret;
        },
        __wbg_code_01dd0bd9fcc03959: function(arg0, arg1) {
            const ret = arg1.code;
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg_colorMask_5f10e62af55cf3df: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.colorMask(arg1 !== 0, arg2 !== 0, arg3 !== 0, arg4 !== 0);
        },
        __wbg_colorMask_928a1d1549f0b649: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.colorMask(arg1 !== 0, arg2 !== 0, arg3 !== 0, arg4 !== 0);
        },
        __wbg_compileShader_53017c42b0ebefd1: function(arg0, arg1) {
            arg0.compileShader(arg1);
        },
        __wbg_compileShader_829e0d2e1845103a: function(arg0, arg1) {
            arg0.compileShader(arg1);
        },
        __wbg_compressedTexSubImage2D_3b1ce5520142cdca: function(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8) {
            arg0.compressedTexSubImage2D(arg1 >>> 0, arg2, arg3, arg4, arg5, arg6, arg7 >>> 0, arg8);
        },
        __wbg_compressedTexSubImage2D_6365572740975d9c: function(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8) {
            arg0.compressedTexSubImage2D(arg1 >>> 0, arg2, arg3, arg4, arg5, arg6, arg7 >>> 0, arg8);
        },
        __wbg_compressedTexSubImage2D_e07bcccfa91f9ded: function(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9) {
            arg0.compressedTexSubImage2D(arg1 >>> 0, arg2, arg3, arg4, arg5, arg6, arg7 >>> 0, arg8, arg9);
        },
        __wbg_compressedTexSubImage3D_684b729638e524b2: function(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11) {
            arg0.compressedTexSubImage3D(arg1 >>> 0, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9 >>> 0, arg10, arg11);
        },
        __wbg_compressedTexSubImage3D_a024007677c0170d: function(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10) {
            arg0.compressedTexSubImage3D(arg1 >>> 0, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9 >>> 0, arg10);
        },
        __wbg_copyBufferSubData_9b77b154f0d617f9: function(arg0, arg1, arg2, arg3, arg4, arg5) {
            arg0.copyBufferSubData(arg1 >>> 0, arg2 >>> 0, arg3, arg4, arg5);
        },
        __wbg_copyTexSubImage2D_868a6e50a76af98c: function(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8) {
            arg0.copyTexSubImage2D(arg1 >>> 0, arg2, arg3, arg4, arg5, arg6, arg7, arg8);
        },
        __wbg_copyTexSubImage2D_b8babe814bc11855: function(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8) {
            arg0.copyTexSubImage2D(arg1 >>> 0, arg2, arg3, arg4, arg5, arg6, arg7, arg8);
        },
        __wbg_copyTexSubImage3D_62c142b3f6732e33: function(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9) {
            arg0.copyTexSubImage3D(arg1 >>> 0, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9);
        },
        __wbg_createBuffer_494c1ef28cb9e6ae: function(arg0) {
            const ret = arg0.createBuffer();
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_createBuffer_f9bfe40a3f9433d1: function(arg0) {
            const ret = arg0.createBuffer();
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_createElement_cb4bc1ab7d348ae0: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = arg0.createElement(getStringFromWasm0(arg1, arg2));
            return ret;
        }, arguments); },
        __wbg_createFramebuffer_a1d26795ba728df5: function(arg0) {
            const ret = arg0.createFramebuffer();
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_createFramebuffer_a7c47212148ba52c: function(arg0) {
            const ret = arg0.createFramebuffer();
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_createObjectURL_14df957177414818: function() { return handleError(function (arg0, arg1) {
            const ret = URL.createObjectURL(arg1);
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        }, arguments); },
        __wbg_createProgram_91311433af129f6b: function(arg0) {
            const ret = arg0.createProgram();
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_createProgram_9c6810f61c2867e8: function(arg0) {
            const ret = arg0.createProgram();
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_createQuery_a59efd2d0926b033: function(arg0) {
            const ret = arg0.createQuery();
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_createRenderbuffer_7f63361f79a0c6d6: function(arg0) {
            const ret = arg0.createRenderbuffer();
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_createRenderbuffer_b30eb74d22747f16: function(arg0) {
            const ret = arg0.createRenderbuffer();
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_createSampler_c065edefc55bf7c5: function(arg0) {
            const ret = arg0.createSampler();
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_createShader_16e1435751f9810b: function(arg0, arg1) {
            const ret = arg0.createShader(arg1 >>> 0);
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_createShader_d7bde1e12aaee783: function(arg0, arg1) {
            const ret = arg0.createShader(arg1 >>> 0);
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_createTexture_6cd54bf5baa96fea: function(arg0) {
            const ret = arg0.createTexture();
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_createTexture_d3e94ea49eb0da8d: function(arg0) {
            const ret = arg0.createTexture();
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_createVertexArrayOES_2725ef0c78c36fdb: function(arg0) {
            const ret = arg0.createVertexArrayOES();
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_createVertexArray_c832d09ef439799f: function(arg0) {
            const ret = arg0.createVertexArray();
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_ctrlKey_39014bf3626845f5: function(arg0) {
            const ret = arg0.ctrlKey;
            return ret;
        },
        __wbg_ctrlKey_ac2366c122f9e622: function(arg0) {
            const ret = arg0.ctrlKey;
            return ret;
        },
        __wbg_cullFace_36165665a0503d57: function(arg0, arg1) {
            arg0.cullFace(arg1 >>> 0);
        },
        __wbg_cullFace_52aca0bbeed3ca12: function(arg0, arg1) {
            arg0.cullFace(arg1 >>> 0);
        },
        __wbg_data_d26f5825b8c5890f: function(arg0) {
            const ret = arg0.data;
            return ret;
        },
        __wbg_debug_1f40aa169fc5969a: function(arg0) {
            console.debug(arg0);
        },
        __wbg_deleteBuffer_bc26013c1349dcbe: function(arg0, arg1) {
            arg0.deleteBuffer(arg1);
        },
        __wbg_deleteBuffer_e53134870573adc2: function(arg0, arg1) {
            arg0.deleteBuffer(arg1);
        },
        __wbg_deleteFramebuffer_e5bad11871de4b0b: function(arg0, arg1) {
            arg0.deleteFramebuffer(arg1);
        },
        __wbg_deleteFramebuffer_f0a2a7e6268a8543: function(arg0, arg1) {
            arg0.deleteFramebuffer(arg1);
        },
        __wbg_deleteProgram_3b5765b7ba778323: function(arg0, arg1) {
            arg0.deleteProgram(arg1);
        },
        __wbg_deleteProgram_8da9310dc3e3dc4a: function(arg0, arg1) {
            arg0.deleteProgram(arg1);
        },
        __wbg_deleteQuery_737cac127ab2dc78: function(arg0, arg1) {
            arg0.deleteQuery(arg1);
        },
        __wbg_deleteRenderbuffer_ad50e9f8d6048e44: function(arg0, arg1) {
            arg0.deleteRenderbuffer(arg1);
        },
        __wbg_deleteRenderbuffer_c1d5f908f13cbc4a: function(arg0, arg1) {
            arg0.deleteRenderbuffer(arg1);
        },
        __wbg_deleteSampler_04ff5350f3bbb797: function(arg0, arg1) {
            arg0.deleteSampler(arg1);
        },
        __wbg_deleteShader_2356d1c31b1783cd: function(arg0, arg1) {
            arg0.deleteShader(arg1);
        },
        __wbg_deleteShader_fb2bc9b8173879a2: function(arg0, arg1) {
            arg0.deleteShader(arg1);
        },
        __wbg_deleteSync_0e6550212cb708de: function(arg0, arg1) {
            arg0.deleteSync(arg1);
        },
        __wbg_deleteTexture_4409d07af42e9f38: function(arg0, arg1) {
            arg0.deleteTexture(arg1);
        },
        __wbg_deleteTexture_bbc4c70dc6771f06: function(arg0, arg1) {
            arg0.deleteTexture(arg1);
        },
        __wbg_deleteVertexArrayOES_8e0816a94bce26fb: function(arg0, arg1) {
            arg0.deleteVertexArrayOES(arg1);
        },
        __wbg_deleteVertexArray_f55f4868be717395: function(arg0, arg1) {
            arg0.deleteVertexArray(arg1);
        },
        __wbg_deltaMode_451fea8887a9d7dd: function(arg0) {
            const ret = arg0.deltaMode;
            return ret;
        },
        __wbg_deltaX_92160bde4f980bd0: function(arg0) {
            const ret = arg0.deltaX;
            return ret;
        },
        __wbg_deltaY_081ac8dff566df4c: function(arg0) {
            const ret = arg0.deltaY;
            return ret;
        },
        __wbg_depthFunc_77bdfe07e7010de2: function(arg0, arg1) {
            arg0.depthFunc(arg1 >>> 0);
        },
        __wbg_depthFunc_87ae202aa94cddf3: function(arg0, arg1) {
            arg0.depthFunc(arg1 >>> 0);
        },
        __wbg_depthMask_061e43ee55aed14a: function(arg0, arg1) {
            arg0.depthMask(arg1 !== 0);
        },
        __wbg_depthMask_c6dfa790778c278d: function(arg0, arg1) {
            arg0.depthMask(arg1 !== 0);
        },
        __wbg_depthRange_ae3428ff646c9628: function(arg0, arg1, arg2) {
            arg0.depthRange(arg1, arg2);
        },
        __wbg_depthRange_ddc7d8c37efb7917: function(arg0, arg1, arg2) {
            arg0.depthRange(arg1, arg2);
        },
        __wbg_devicePixelRatio_3b16e3d7bfeb3d74: function(arg0) {
            const ret = arg0.devicePixelRatio;
            return ret;
        },
        __wbg_disableVertexAttribArray_8a9e21c07b023518: function(arg0, arg1) {
            arg0.disableVertexAttribArray(arg1 >>> 0);
        },
        __wbg_disableVertexAttribArray_fbe7fab26f2a71c2: function(arg0, arg1) {
            arg0.disableVertexAttribArray(arg1 >>> 0);
        },
        __wbg_disable_19259de11dc0ae6a: function(arg0, arg1) {
            arg0.disable(arg1 >>> 0);
        },
        __wbg_disable_6c8639d7dd6759b0: function(arg0, arg1) {
            arg0.disable(arg1 >>> 0);
        },
        __wbg_document_dea51c4e8b7e2f92: function(arg0) {
            const ret = arg0.document;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_drawArraysInstancedANGLE_7b1b2480628b8083: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.drawArraysInstancedANGLE(arg1 >>> 0, arg2, arg3, arg4);
        },
        __wbg_drawArraysInstanced_814213d5ef89f7a8: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.drawArraysInstanced(arg1 >>> 0, arg2, arg3, arg4);
        },
        __wbg_drawArrays_26e8c1ffa1fae210: function(arg0, arg1, arg2, arg3) {
            arg0.drawArrays(arg1 >>> 0, arg2, arg3);
        },
        __wbg_drawArrays_40d08c28945a6f65: function(arg0, arg1, arg2, arg3) {
            arg0.drawArrays(arg1 >>> 0, arg2, arg3);
        },
        __wbg_drawBuffersWEBGL_6af877b7408abd60: function(arg0, arg1) {
            arg0.drawBuffersWEBGL(arg1);
        },
        __wbg_drawBuffers_2796956da73e9d9d: function(arg0, arg1) {
            arg0.drawBuffers(arg1);
        },
        __wbg_drawElementsInstancedANGLE_4197b7d965687804: function(arg0, arg1, arg2, arg3, arg4, arg5) {
            arg0.drawElementsInstancedANGLE(arg1 >>> 0, arg2, arg3 >>> 0, arg4, arg5);
        },
        __wbg_drawElementsInstanced_f87ae3f3cca31fab: function(arg0, arg1, arg2, arg3, arg4, arg5) {
            arg0.drawElementsInstanced(arg1 >>> 0, arg2, arg3 >>> 0, arg4, arg5);
        },
        __wbg_enableVertexAttribArray_1487529724c4367e: function(arg0, arg1) {
            arg0.enableVertexAttribArray(arg1 >>> 0);
        },
        __wbg_enableVertexAttribArray_51df768fe987ab9d: function(arg0, arg1) {
            arg0.enableVertexAttribArray(arg1 >>> 0);
        },
        __wbg_enable_24c93e77c2f36674: function(arg0, arg1) {
            arg0.enable(arg1 >>> 0);
        },
        __wbg_enable_ff85c188ec9d52c4: function(arg0, arg1) {
            arg0.enable(arg1 >>> 0);
        },
        __wbg_endQuery_3b7ffb399c262d9c: function(arg0, arg1) {
            arg0.endQuery(arg1 >>> 0);
        },
        __wbg_error_1fe34cc606d34980: function(arg0, arg1) {
            console.error(arg0, arg1);
        },
        __wbg_error_7bbe8dfe14e44fdd: function(arg0) {
            console.error(arg0);
        },
        __wbg_error_a6fa202b58aa1cd3: function(arg0, arg1) {
            let deferred0_0;
            let deferred0_1;
            try {
                deferred0_0 = arg0;
                deferred0_1 = arg1;
                console.error(getStringFromWasm0(arg0, arg1));
            } finally {
                wasm.__wbindgen_free(deferred0_0, deferred0_1, 1);
            }
        },
        __wbg_fenceSync_0f02caf5ba67c223: function(arg0, arg1, arg2) {
            const ret = arg0.fenceSync(arg1 >>> 0, arg2 >>> 0);
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_framebufferRenderbuffer_24c4d62cd8e6b4cd: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.framebufferRenderbuffer(arg1 >>> 0, arg2 >>> 0, arg3 >>> 0, arg4);
        },
        __wbg_framebufferRenderbuffer_3ad7151a5c9187b8: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.framebufferRenderbuffer(arg1 >>> 0, arg2 >>> 0, arg3 >>> 0, arg4);
        },
        __wbg_framebufferTexture2D_80f786580439279e: function(arg0, arg1, arg2, arg3, arg4, arg5) {
            arg0.framebufferTexture2D(arg1 >>> 0, arg2 >>> 0, arg3 >>> 0, arg4, arg5);
        },
        __wbg_framebufferTexture2D_a2e2d4802f0e375a: function(arg0, arg1, arg2, arg3, arg4, arg5) {
            arg0.framebufferTexture2D(arg1 >>> 0, arg2 >>> 0, arg3 >>> 0, arg4, arg5);
        },
        __wbg_framebufferTextureLayer_011ed0df2e8b357f: function(arg0, arg1, arg2, arg3, arg4, arg5) {
            arg0.framebufferTextureLayer(arg1 >>> 0, arg2 >>> 0, arg3, arg4, arg5);
        },
        __wbg_framebufferTextureMultiviewOVR_e9cff91bfdca55a1: function(arg0, arg1, arg2, arg3, arg4, arg5, arg6) {
            arg0.framebufferTextureMultiviewOVR(arg1 >>> 0, arg2 >>> 0, arg3, arg4, arg5, arg6);
        },
        __wbg_from_72058277f2aa1cbb: function(arg0) {
            const ret = Array.from(arg0);
            return ret;
        },
        __wbg_frontFace_3023fb36ed67cc73: function(arg0, arg1) {
            arg0.frontFace(arg1 >>> 0);
        },
        __wbg_frontFace_db64f8ca64bcb512: function(arg0, arg1) {
            arg0.frontFace(arg1 >>> 0);
        },
        __wbg_fullscreenElement_7ab49b8f16149259: function(arg0) {
            const ret = arg0.fullscreenElement;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_getActiveUniform_07bb73f903f5d25b: function(arg0, arg1, arg2) {
            const ret = arg0.getActiveUniform(arg1, arg2 >>> 0);
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_getActiveUniform_8298e18e6bbc2adb: function(arg0, arg1, arg2) {
            const ret = arg0.getActiveUniform(arg1, arg2 >>> 0);
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_getBoundingClientRect_f55da1d1ef6c9525: function(arg0) {
            const ret = arg0.getBoundingClientRect();
            return ret;
        },
        __wbg_getBufferSubData_4956c3b6a88f6371: function(arg0, arg1, arg2, arg3) {
            arg0.getBufferSubData(arg1 >>> 0, arg2, arg3);
        },
        __wbg_getContext_dc7a9741c78723b9: function() { return handleError(function (arg0, arg1, arg2, arg3) {
            const ret = arg0.getContext(getStringFromWasm0(arg1, arg2), arg3);
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        }, arguments); },
        __wbg_getExtension_02317c16e507871c: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = arg0.getExtension(getStringFromWasm0(arg1, arg2));
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        }, arguments); },
        __wbg_getIndexedParameter_83508eb473b3ff62: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = arg0.getIndexedParameter(arg1 >>> 0, arg2 >>> 0);
            return ret;
        }, arguments); },
        __wbg_getModifierState_bf305d3a62b6fc52: function(arg0, arg1, arg2) {
            const ret = arg0.getModifierState(getStringFromWasm0(arg1, arg2));
            return ret;
        },
        __wbg_getParameter_a3b6cf03bb2fc70e: function() { return handleError(function (arg0, arg1) {
            const ret = arg0.getParameter(arg1 >>> 0);
            return ret;
        }, arguments); },
        __wbg_getParameter_edb1fa70b989e0f6: function() { return handleError(function (arg0, arg1) {
            const ret = arg0.getParameter(arg1 >>> 0);
            return ret;
        }, arguments); },
        __wbg_getProgramInfoLog_3a5d46d465034b75: function(arg0, arg1, arg2) {
            const ret = arg1.getProgramInfoLog(arg2);
            var ptr1 = isLikeNone(ret) ? 0 : passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            var len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg_getProgramInfoLog_5962dcdfa36832e6: function(arg0, arg1, arg2) {
            const ret = arg1.getProgramInfoLog(arg2);
            var ptr1 = isLikeNone(ret) ? 0 : passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            var len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg_getProgramParameter_78d1cde9dde84c4f: function(arg0, arg1, arg2) {
            const ret = arg0.getProgramParameter(arg1, arg2 >>> 0);
            return ret;
        },
        __wbg_getProgramParameter_a42dd81e20e391fb: function(arg0, arg1, arg2) {
            const ret = arg0.getProgramParameter(arg1, arg2 >>> 0);
            return ret;
        },
        __wbg_getQueryParameter_bccf5fb396a361fe: function(arg0, arg1, arg2) {
            const ret = arg0.getQueryParameter(arg1, arg2 >>> 0);
            return ret;
        },
        __wbg_getShaderInfoLog_2ed5dd00cbdcad8b: function(arg0, arg1, arg2) {
            const ret = arg1.getShaderInfoLog(arg2);
            var ptr1 = isLikeNone(ret) ? 0 : passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            var len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg_getShaderInfoLog_6bea42e579c265c9: function(arg0, arg1, arg2) {
            const ret = arg1.getShaderInfoLog(arg2);
            var ptr1 = isLikeNone(ret) ? 0 : passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            var len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg_getShaderParameter_e26235fabda9d679: function(arg0, arg1, arg2) {
            const ret = arg0.getShaderParameter(arg1, arg2 >>> 0);
            return ret;
        },
        __wbg_getShaderParameter_f04d47df4c913f81: function(arg0, arg1, arg2) {
            const ret = arg0.getShaderParameter(arg1, arg2 >>> 0);
            return ret;
        },
        __wbg_getSupportedExtensions_0e7a559d88edfa2c: function(arg0) {
            const ret = arg0.getSupportedExtensions();
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_getSyncParameter_f535005fc5d8096e: function(arg0, arg1, arg2) {
            const ret = arg0.getSyncParameter(arg1, arg2 >>> 0);
            return ret;
        },
        __wbg_getUniformBlockIndex_a83a13830d227cbf: function(arg0, arg1, arg2, arg3) {
            const ret = arg0.getUniformBlockIndex(arg1, getStringFromWasm0(arg2, arg3));
            return ret;
        },
        __wbg_getUniformLocation_169e03b8181c4a18: function(arg0, arg1, arg2, arg3) {
            const ret = arg0.getUniformLocation(arg1, getStringFromWasm0(arg2, arg3));
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_getUniformLocation_f6343037d8fd2e48: function(arg0, arg1, arg2, arg3) {
            const ret = arg0.getUniformLocation(arg1, getStringFromWasm0(arg2, arg3));
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_get_37e3667d4ef7fd02: function(arg0, arg1) {
            const ret = arg0[arg1 >>> 0];
            return ret;
        },
        __wbg_get_50a442d113ca286d: function(arg0, arg1, arg2) {
            const ret = arg0[getStringFromWasm0(arg1, arg2)];
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_get_c5244f6b1b253da9: function() { return handleError(function (arg0, arg1) {
            const ret = Reflect.get(arg0, arg1);
            return ret;
        }, arguments); },
        __wbg_globalThis_6d268067835e6709: function() { return handleError(function () {
            const ret = globalThis.globalThis;
            return ret;
        }, arguments); },
        __wbg_global_3fe6c6c8ad6e6fb2: function() { return handleError(function () {
            const ret = global.global;
            return ret;
        }, arguments); },
        __wbg_height_2023c39d3117bb97: function(arg0) {
            const ret = arg0.height;
            return ret;
        },
        __wbg_height_7267a0e519f6dba1: function(arg0) {
            const ret = arg0.height;
            return ret;
        },
        __wbg_height_8dfcad8fe2cce17a: function(arg0) {
            const ret = arg0.height;
            return ret;
        },
        __wbg_href_5c0aae2580cca5d3: function(arg0, arg1) {
            const ret = arg1.href;
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg_href_f8c300984d66649e: function() { return handleError(function (arg0, arg1) {
            const ret = arg1.href;
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        }, arguments); },
        __wbg_info_aca1010228a0c7fb: function(arg0) {
            console.info(arg0);
        },
        __wbg_innerHeight_1cb32c4e90b03397: function() { return handleError(function (arg0) {
            const ret = arg0.innerHeight;
            return ret;
        }, arguments); },
        __wbg_innerWidth_b0cba3cc62bd4f07: function() { return handleError(function (arg0) {
            const ret = arg0.innerWidth;
            return ret;
        }, arguments); },
        __wbg_instanceof_HtmlCanvasElement_c48303f45c36353c: function(arg0) {
            let result;
            try {
                result = arg0 instanceof HTMLCanvasElement;
            } catch (_) {
                result = false;
            }
            const ret = result;
            return ret;
        },
        __wbg_instanceof_WebGl2RenderingContext_6dc6109db8c6889e: function(arg0) {
            let result;
            try {
                result = arg0 instanceof WebGL2RenderingContext;
            } catch (_) {
                result = false;
            }
            const ret = result;
            return ret;
        },
        __wbg_instanceof_Window_b6121c2f80910d3f: function(arg0) {
            let result;
            try {
                result = arg0 instanceof Window;
            } catch (_) {
                result = false;
            }
            const ret = result;
            return ret;
        },
        __wbg_invalidateFramebuffer_097b5854dbd7bac8: function() { return handleError(function (arg0, arg1, arg2) {
            arg0.invalidateFramebuffer(arg1 >>> 0, arg2);
        }, arguments); },
        __wbg_is_47683e527415e280: function(arg0, arg1) {
            const ret = Object.is(arg0, arg1);
            return ret;
        },
        __wbg_keyCode_99c2aca8c357b4cf: function(arg0) {
            const ret = arg0.keyCode;
            return ret;
        },
        __wbg_key_4cc999324d79695e: function(arg0, arg1) {
            const ret = arg1.key;
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg_length_1e876741fdb53aee: function(arg0) {
            const ret = arg0.length;
            return ret;
        },
        __wbg_linkProgram_5cf1789a9b0a6290: function(arg0, arg1) {
            arg0.linkProgram(arg1);
        },
        __wbg_linkProgram_e8dc07e7109cbea4: function(arg0, arg1) {
            arg0.linkProgram(arg1);
        },
        __wbg_location_14b9ac40c23935db: function(arg0) {
            const ret = arg0.location;
            return ret;
        },
        __wbg_log_12036ba238c756ad: function(arg0) {
            console.log(arg0);
        },
        __wbg_matchMedia_8dff30aad9ac616b: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = arg0.matchMedia(getStringFromWasm0(arg1, arg2));
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        }, arguments); },
        __wbg_matches_28c292b0a1ae86ff: function(arg0) {
            const ret = arg0.matches;
            return ret;
        },
        __wbg_matches_3f95671b71612bd8: function(arg0) {
            const ret = arg0.matches;
            return ret;
        },
        __wbg_metaKey_3ad95931c3d00e42: function(arg0) {
            const ret = arg0.metaKey;
            return ret;
        },
        __wbg_metaKey_5f4c9d8c09cbbe5b: function(arg0) {
            const ret = arg0.metaKey;
            return ret;
        },
        __wbg_movementX_7d667d714f4c40dd: function(arg0) {
            const ret = arg0.movementX;
            return ret;
        },
        __wbg_movementY_d4fe9c431df79108: function(arg0) {
            const ret = arg0.movementY;
            return ret;
        },
        __wbg_name_e1e4972b2016f290: function(arg0, arg1) {
            const ret = arg1.name;
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg_new_227d7c05414eb861: function() {
            const ret = new Error();
            return ret;
        },
        __wbg_new_364c96143b8f3496: function() {
            const ret = new Object();
            return ret;
        },
        __wbg_new_d9762fd75876aafe: function() {
            const ret = new Array();
            return ret;
        },
        __wbg_new_no_args_4010ad257320fa4f: function(arg0, arg1) {
            const ret = new Function(getStringFromWasm0(arg0, arg1));
            return ret;
        },
        __wbg_new_with_base_e03acfa85a61b469: function() { return handleError(function (arg0, arg1, arg2, arg3) {
            const ret = new URL(getStringFromWasm0(arg0, arg1), getStringFromWasm0(arg2, arg3));
            return ret;
        }, arguments); },
        __wbg_new_with_byte_offset_and_length_1a5962804d390af1: function(arg0, arg1, arg2) {
            const ret = new Float32Array(arg0, arg1 >>> 0, arg2 >>> 0);
            return ret;
        },
        __wbg_new_with_byte_offset_and_length_348cae7153302d55: function(arg0, arg1, arg2) {
            const ret = new Int8Array(arg0, arg1 >>> 0, arg2 >>> 0);
            return ret;
        },
        __wbg_new_with_byte_offset_and_length_4fa6e003670f6dd2: function(arg0, arg1, arg2) {
            const ret = new Int16Array(arg0, arg1 >>> 0, arg2 >>> 0);
            return ret;
        },
        __wbg_new_with_byte_offset_and_length_6d3c5bb1219aa637: function(arg0, arg1, arg2) {
            const ret = new Int32Array(arg0, arg1 >>> 0, arg2 >>> 0);
            return ret;
        },
        __wbg_new_with_byte_offset_and_length_703e75abe9ad6f1b: function(arg0, arg1, arg2) {
            const ret = new Uint16Array(arg0, arg1 >>> 0, arg2 >>> 0);
            return ret;
        },
        __wbg_new_with_byte_offset_and_length_8b21e3b1308deb48: function(arg0, arg1, arg2) {
            const ret = new Uint8Array(arg0, arg1 >>> 0, arg2 >>> 0);
            return ret;
        },
        __wbg_new_with_byte_offset_and_length_c16881e625ffcf92: function(arg0, arg1, arg2) {
            const ret = new Uint32Array(arg0, arg1 >>> 0, arg2 >>> 0);
            return ret;
        },
        __wbg_new_with_options_d0e72d42f8d34dfa: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = new Worker(getStringFromWasm0(arg0, arg1), arg2);
            return ret;
        }, arguments); },
        __wbg_new_with_str_sequence_and_options_bc9d48f2712d547a: function() { return handleError(function (arg0, arg1) {
            const ret = new Blob(arg0, arg1);
            return ret;
        }, arguments); },
        __wbg_now_09297098ebcedb6d: function() {
            const ret = Date.now();
            return ret;
        },
        __wbg_now_4145b9f28216dc49: function(arg0) {
            const ret = arg0.now();
            return ret;
        },
        __wbg_of_3d16f54772ac7eeb: function(arg0) {
            const ret = Array.of(arg0);
            return ret;
        },
        __wbg_offsetX_25d094312e0f403a: function(arg0) {
            const ret = arg0.offsetX;
            return ret;
        },
        __wbg_offsetY_7721fb558a0f0924: function(arg0) {
            const ret = arg0.offsetY;
            return ret;
        },
        __wbg_pixelStorei_2f59397b82397a10: function(arg0, arg1, arg2) {
            arg0.pixelStorei(arg1 >>> 0, arg2);
        },
        __wbg_pixelStorei_858fcdc53fc6d5f7: function(arg0, arg1, arg2) {
            arg0.pixelStorei(arg1 >>> 0, arg2);
        },
        __wbg_pointerId_7fadaccac5328f88: function(arg0) {
            const ret = arg0.pointerId;
            return ret;
        },
        __wbg_polygonOffset_10d01c70fe60c868: function(arg0, arg1, arg2) {
            arg0.polygonOffset(arg1, arg2);
        },
        __wbg_polygonOffset_4722c971930800a4: function(arg0, arg1, arg2) {
            arg0.polygonOffset(arg1, arg2);
        },
        __wbg_postMessage_9ae19589033fb9c6: function() { return handleError(function (arg0, arg1) {
            arg0.postMessage(arg1);
        }, arguments); },
        __wbg_preventDefault_3831269806069e1e: function(arg0) {
            arg0.preventDefault();
        },
        __wbg_push_1303ce035391aed3: function(arg0, arg1) {
            const ret = arg0.push(arg1);
            return ret;
        },
        __wbg_querySelector_7169320ac864e16b: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = arg0.querySelector(getStringFromWasm0(arg1, arg2));
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        }, arguments); },
        __wbg_readBuffer_12832bb7195f9bcb: function(arg0, arg1) {
            arg0.readBuffer(arg1 >>> 0);
        },
        __wbg_readPixels_658faa18625d2922: function() { return handleError(function (arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7) {
            arg0.readPixels(arg1, arg2, arg3, arg4, arg5 >>> 0, arg6 >>> 0, arg7);
        }, arguments); },
        __wbg_readPixels_9b9461727ad22739: function() { return handleError(function (arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7) {
            arg0.readPixels(arg1, arg2, arg3, arg4, arg5 >>> 0, arg6 >>> 0, arg7);
        }, arguments); },
        __wbg_readPixels_de90d819d7bfc548: function() { return handleError(function (arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7) {
            arg0.readPixels(arg1, arg2, arg3, arg4, arg5 >>> 0, arg6 >>> 0, arg7);
        }, arguments); },
        __wbg_removeEventListener_28750678b6c263d2: function() { return handleError(function (arg0, arg1, arg2, arg3) {
            arg0.removeEventListener(getStringFromWasm0(arg1, arg2), arg3);
        }, arguments); },
        __wbg_removeListener_c4b51deef0e5097f: function() { return handleError(function (arg0, arg1) {
            arg0.removeListener(arg1);
        }, arguments); },
        __wbg_renderbufferStorageMultisample_737f221eb8f2179a: function(arg0, arg1, arg2, arg3, arg4, arg5) {
            arg0.renderbufferStorageMultisample(arg1 >>> 0, arg2, arg3 >>> 0, arg4, arg5);
        },
        __wbg_renderbufferStorage_83349d54a691992e: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.renderbufferStorage(arg1 >>> 0, arg2 >>> 0, arg3, arg4);
        },
        __wbg_renderbufferStorage_a7faaf1433132c3e: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.renderbufferStorage(arg1 >>> 0, arg2 >>> 0, arg3, arg4);
        },
        __wbg_requestAnimationFrame_b329e06df3659010: function() { return handleError(function (arg0, arg1) {
            const ret = arg0.requestAnimationFrame(arg1);
            return ret;
        }, arguments); },
        __wbg_requestFullscreen_35f64e1b96b7667e: function() { return handleError(function (arg0) {
            arg0.requestFullscreen();
        }, arguments); },
        __wbg_resolve_f6e1e4e0c9251963: function(arg0) {
            const ret = Promise.resolve(arg0);
            return ret;
        },
        __wbg_revokeObjectURL_f6bf8117ccfbee0b: function() { return handleError(function (arg0, arg1) {
            URL.revokeObjectURL(getStringFromWasm0(arg0, arg1));
        }, arguments); },
        __wbg_samplerParameterf_1644d663f597a8b2: function(arg0, arg1, arg2, arg3) {
            arg0.samplerParameterf(arg1, arg2 >>> 0, arg3);
        },
        __wbg_samplerParameteri_edb9d1f9607553ed: function(arg0, arg1, arg2, arg3) {
            arg0.samplerParameteri(arg1, arg2 >>> 0, arg3);
        },
        __wbg_scissor_932c158cd1e15f61: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.scissor(arg1, arg2, arg3, arg4);
        },
        __wbg_scissor_f8286688f79b5930: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.scissor(arg1, arg2, arg3, arg4);
        },
        __wbg_self_1035a7cbd1b0d959: function() { return handleError(function () {
            const ret = self.self;
            return ret;
        }, arguments); },
        __wbg_setAttribute_31a29574ab0af743: function() { return handleError(function (arg0, arg1, arg2, arg3, arg4) {
            arg0.setAttribute(getStringFromWasm0(arg1, arg2), getStringFromWasm0(arg3, arg4));
        }, arguments); },
        __wbg_setPointerCapture_f1ef5c9eb0f2cffd: function() { return handleError(function (arg0, arg1) {
            arg0.setPointerCapture(arg1);
        }, arguments); },
        __wbg_setProperty_c9351acebf844838: function() { return handleError(function (arg0, arg1, arg2, arg3, arg4) {
            arg0.setProperty(getStringFromWasm0(arg1, arg2), getStringFromWasm0(arg3, arg4));
        }, arguments); },
        __wbg_setTimeout_1f3fd0985278756b: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = arg0.setTimeout(arg1, arg2);
            return ret;
        }, arguments); },
        __wbg_set_1ddc4b8cd44d0da4: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = Reflect.set(arg0, arg1, arg2);
            return ret;
        }, arguments); },
        __wbg_set_height_3bcea85d98dbe27e: function(arg0, arg1) {
            arg0.height = arg1 >>> 0;
        },
        __wbg_set_height_9e1a9275e92b1d41: function(arg0, arg1) {
            arg0.height = arg1 >>> 0;
        },
        __wbg_set_onerror_6d54e5733b181da2: function(arg0, arg1) {
            arg0.onerror = arg1;
        },
        __wbg_set_onmessage_dfd01ddf3c7df5ae: function(arg0, arg1) {
            arg0.onmessage = arg1;
        },
        __wbg_set_onmessageerror_bbe783fac83dc607: function(arg0, arg1) {
            arg0.onmessageerror = arg1;
        },
        __wbg_set_width_22ca8a681db59658: function(arg0, arg1) {
            arg0.width = arg1 >>> 0;
        },
        __wbg_set_width_d0f6152108ca5d94: function(arg0, arg1) {
            arg0.width = arg1 >>> 0;
        },
        __wbg_shaderSource_44fd52215d922e99: function(arg0, arg1, arg2, arg3) {
            arg0.shaderSource(arg1, getStringFromWasm0(arg2, arg3));
        },
        __wbg_shaderSource_654d43f847d42673: function(arg0, arg1, arg2, arg3) {
            arg0.shaderSource(arg1, getStringFromWasm0(arg2, arg3));
        },
        __wbg_shiftKey_7a63daf6bbe619fc: function(arg0) {
            const ret = arg0.shiftKey;
            return ret;
        },
        __wbg_shiftKey_b93489105827d938: function(arg0) {
            const ret = arg0.shiftKey;
            return ret;
        },
        __wbg_size_cff4fce88696f6ce: function(arg0) {
            const ret = arg0.size;
            return ret;
        },
        __wbg_stack_3b0d974bbf31e44f: function(arg0, arg1) {
            const ret = arg1.stack;
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg_stencilFuncSeparate_31ef2a945f6a659c: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.stencilFuncSeparate(arg1 >>> 0, arg2 >>> 0, arg3, arg4 >>> 0);
        },
        __wbg_stencilFuncSeparate_c3e04d79366f74d9: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.stencilFuncSeparate(arg1 >>> 0, arg2 >>> 0, arg3, arg4 >>> 0);
        },
        __wbg_stencilMaskSeparate_50e4f46d5fac73b6: function(arg0, arg1, arg2) {
            arg0.stencilMaskSeparate(arg1 >>> 0, arg2 >>> 0);
        },
        __wbg_stencilMaskSeparate_ee0e44bfd85a046c: function(arg0, arg1, arg2) {
            arg0.stencilMaskSeparate(arg1 >>> 0, arg2 >>> 0);
        },
        __wbg_stencilMask_9ca8d5cf4854f74b: function(arg0, arg1) {
            arg0.stencilMask(arg1 >>> 0);
        },
        __wbg_stencilMask_d3c06a01f9962307: function(arg0, arg1) {
            arg0.stencilMask(arg1 >>> 0);
        },
        __wbg_stencilOpSeparate_36e6bf40f8486dae: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.stencilOpSeparate(arg1 >>> 0, arg2 >>> 0, arg3 >>> 0, arg4 >>> 0);
        },
        __wbg_stencilOpSeparate_5b4527b3de05d83b: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.stencilOpSeparate(arg1 >>> 0, arg2 >>> 0, arg3 >>> 0, arg4 >>> 0);
        },
        __wbg_stopPropagation_e9dda3795e897a07: function(arg0) {
            arg0.stopPropagation();
        },
        __wbg_style_d9d65b637c856143: function(arg0) {
            const ret = arg0.style;
            return ret;
        },
        __wbg_target_d579e98a98c929a3: function(arg0) {
            const ret = arg0.target;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_terminate_2fa6a37bf361e08d: function(arg0) {
            arg0.terminate();
        },
        __wbg_texParameteri_abcf9def98c0ddbd: function(arg0, arg1, arg2, arg3) {
            arg0.texParameteri(arg1 >>> 0, arg2 >>> 0, arg3);
        },
        __wbg_texParameteri_bf2482d158723fd2: function(arg0, arg1, arg2, arg3) {
            arg0.texParameteri(arg1 >>> 0, arg2 >>> 0, arg3);
        },
        __wbg_texStorage2D_711353f623731ef8: function(arg0, arg1, arg2, arg3, arg4, arg5) {
            arg0.texStorage2D(arg1 >>> 0, arg2, arg3 >>> 0, arg4, arg5);
        },
        __wbg_texStorage3D_abb3d730a2f61f5e: function(arg0, arg1, arg2, arg3, arg4, arg5, arg6) {
            arg0.texStorage3D(arg1 >>> 0, arg2, arg3 >>> 0, arg4, arg5, arg6);
        },
        __wbg_texSubImage2D_0db4b8f090649743: function() { return handleError(function (arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9) {
            arg0.texSubImage2D(arg1 >>> 0, arg2, arg3, arg4, arg5, arg6, arg7 >>> 0, arg8 >>> 0, arg9);
        }, arguments); },
        __wbg_texSubImage2D_9ea00b1837001d4f: function() { return handleError(function (arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9) {
            arg0.texSubImage2D(arg1 >>> 0, arg2, arg3, arg4, arg5, arg6, arg7 >>> 0, arg8 >>> 0, arg9);
        }, arguments); },
        __wbg_texSubImage2D_d102bad73d599f4e: function() { return handleError(function (arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9) {
            arg0.texSubImage2D(arg1 >>> 0, arg2, arg3, arg4, arg5, arg6, arg7 >>> 0, arg8 >>> 0, arg9);
        }, arguments); },
        __wbg_texSubImage2D_e5f9ae3ff4e3cf39: function() { return handleError(function (arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9) {
            arg0.texSubImage2D(arg1 >>> 0, arg2, arg3, arg4, arg5, arg6, arg7 >>> 0, arg8 >>> 0, arg9);
        }, arguments); },
        __wbg_texSubImage2D_f3e85073359676ad: function() { return handleError(function (arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9) {
            arg0.texSubImage2D(arg1 >>> 0, arg2, arg3, arg4, arg5, arg6, arg7 >>> 0, arg8 >>> 0, arg9);
        }, arguments); },
        __wbg_texSubImage2D_f66f23377be7872d: function() { return handleError(function (arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9) {
            arg0.texSubImage2D(arg1 >>> 0, arg2, arg3, arg4, arg5, arg6, arg7 >>> 0, arg8 >>> 0, arg9);
        }, arguments); },
        __wbg_texSubImage3D_28d0e70c3d54cfc3: function() { return handleError(function (arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11) {
            arg0.texSubImage3D(arg1 >>> 0, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9 >>> 0, arg10 >>> 0, arg11);
        }, arguments); },
        __wbg_texSubImage3D_2e593bf85ee43437: function() { return handleError(function (arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11) {
            arg0.texSubImage3D(arg1 >>> 0, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9 >>> 0, arg10 >>> 0, arg11);
        }, arguments); },
        __wbg_texSubImage3D_3ec4c1b2288dce59: function() { return handleError(function (arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11) {
            arg0.texSubImage3D(arg1 >>> 0, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9 >>> 0, arg10 >>> 0, arg11);
        }, arguments); },
        __wbg_texSubImage3D_490ecc9150c8ca97: function() { return handleError(function (arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11) {
            arg0.texSubImage3D(arg1 >>> 0, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9 >>> 0, arg10 >>> 0, arg11);
        }, arguments); },
        __wbg_texSubImage3D_dccd717ec2e95e27: function() { return handleError(function (arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11) {
            arg0.texSubImage3D(arg1 >>> 0, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9 >>> 0, arg10 >>> 0, arg11);
        }, arguments); },
        __wbg_then_dd9a54440c7d74fa: function(arg0, arg1) {
            const ret = arg0.then(arg1);
            return ret;
        },
        __wbg_type_be0b5b46c4f3a18a: function(arg0) {
            const ret = arg0.type;
            return ret;
        },
        __wbg_type_df85fdfb6b827f79: function(arg0, arg1) {
            const ret = arg1.type;
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg_uniform1f_89f744e652225dc3: function(arg0, arg1, arg2) {
            arg0.uniform1f(arg1, arg2);
        },
        __wbg_uniform1f_cebaf1ab5e55dc62: function(arg0, arg1, arg2) {
            arg0.uniform1f(arg1, arg2);
        },
        __wbg_uniform1i_2787bcb98fe9f019: function(arg0, arg1, arg2) {
            arg0.uniform1i(arg1, arg2);
        },
        __wbg_uniform1i_c2f58d766dad35f6: function(arg0, arg1, arg2) {
            arg0.uniform1i(arg1, arg2);
        },
        __wbg_uniform2fv_8632c6e37707d833: function(arg0, arg1, arg2, arg3) {
            arg0.uniform2fv(arg1, getArrayF32FromWasm0(arg2, arg3));
        },
        __wbg_uniform2fv_8d09261a6b7f00ef: function(arg0, arg1, arg2, arg3) {
            arg0.uniform2fv(arg1, getArrayF32FromWasm0(arg2, arg3));
        },
        __wbg_uniform2iv_6d13400ea42afdd2: function(arg0, arg1, arg2, arg3) {
            arg0.uniform2iv(arg1, getArrayI32FromWasm0(arg2, arg3));
        },
        __wbg_uniform2iv_f8efc61bb1121917: function(arg0, arg1, arg2, arg3) {
            arg0.uniform2iv(arg1, getArrayI32FromWasm0(arg2, arg3));
        },
        __wbg_uniform3fv_8a73e7c8e24e0a0b: function(arg0, arg1, arg2, arg3) {
            arg0.uniform3fv(arg1, getArrayF32FromWasm0(arg2, arg3));
        },
        __wbg_uniform3fv_e2659f269ea0ddaa: function(arg0, arg1, arg2, arg3) {
            arg0.uniform3fv(arg1, getArrayF32FromWasm0(arg2, arg3));
        },
        __wbg_uniform3iv_a523f87765007414: function(arg0, arg1, arg2, arg3) {
            arg0.uniform3iv(arg1, getArrayI32FromWasm0(arg2, arg3));
        },
        __wbg_uniform3iv_c2d61de6e157a4f5: function(arg0, arg1, arg2, arg3) {
            arg0.uniform3iv(arg1, getArrayI32FromWasm0(arg2, arg3));
        },
        __wbg_uniform4f_609e8cdf0931dd84: function(arg0, arg1, arg2, arg3, arg4, arg5) {
            arg0.uniform4f(arg1, arg2, arg3, arg4, arg5);
        },
        __wbg_uniform4f_a9638277c9ff8704: function(arg0, arg1, arg2, arg3, arg4, arg5) {
            arg0.uniform4f(arg1, arg2, arg3, arg4, arg5);
        },
        __wbg_uniform4fv_30db149c2e5a5d30: function(arg0, arg1, arg2, arg3) {
            arg0.uniform4fv(arg1, getArrayF32FromWasm0(arg2, arg3));
        },
        __wbg_uniform4fv_7a0b05e1ca9d928a: function(arg0, arg1, arg2, arg3) {
            arg0.uniform4fv(arg1, getArrayF32FromWasm0(arg2, arg3));
        },
        __wbg_uniform4iv_243ac527ed1b7fd0: function(arg0, arg1, arg2, arg3) {
            arg0.uniform4iv(arg1, getArrayI32FromWasm0(arg2, arg3));
        },
        __wbg_uniform4iv_e48499377a857f35: function(arg0, arg1, arg2, arg3) {
            arg0.uniform4iv(arg1, getArrayI32FromWasm0(arg2, arg3));
        },
        __wbg_uniformBlockBinding_8a3fce9c07096902: function(arg0, arg1, arg2, arg3) {
            arg0.uniformBlockBinding(arg1, arg2 >>> 0, arg3 >>> 0);
        },
        __wbg_uniformMatrix2fv_21ac4917ceeec2ba: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.uniformMatrix2fv(arg1, arg2 !== 0, getArrayF32FromWasm0(arg3, arg4));
        },
        __wbg_uniformMatrix2fv_76eb3a9d2a44493a: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.uniformMatrix2fv(arg1, arg2 !== 0, getArrayF32FromWasm0(arg3, arg4));
        },
        __wbg_uniformMatrix3fv_24ab1ed0e6dbf321: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.uniformMatrix3fv(arg1, arg2 !== 0, getArrayF32FromWasm0(arg3, arg4));
        },
        __wbg_uniformMatrix3fv_6bcb0b2b47805969: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.uniformMatrix3fv(arg1, arg2 !== 0, getArrayF32FromWasm0(arg3, arg4));
        },
        __wbg_uniformMatrix4fv_252904bfe947a55e: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.uniformMatrix4fv(arg1, arg2 !== 0, getArrayF32FromWasm0(arg3, arg4));
        },
        __wbg_uniformMatrix4fv_c718ba530981768a: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.uniformMatrix4fv(arg1, arg2 !== 0, getArrayF32FromWasm0(arg3, arg4));
        },
        __wbg_useProgram_3f097ebfe4608c37: function(arg0, arg1) {
            arg0.useProgram(arg1);
        },
        __wbg_useProgram_610b461b81e10160: function(arg0, arg1) {
            arg0.useProgram(arg1);
        },
        __wbg_vertexAttribDivisorANGLE_50e976fcfcc13180: function(arg0, arg1, arg2) {
            arg0.vertexAttribDivisorANGLE(arg1 >>> 0, arg2 >>> 0);
        },
        __wbg_vertexAttribDivisor_59bf7abb00b14d4d: function(arg0, arg1, arg2) {
            arg0.vertexAttribDivisor(arg1 >>> 0, arg2 >>> 0);
        },
        __wbg_vertexAttribIPointer_52271879f8a86029: function(arg0, arg1, arg2, arg3, arg4, arg5) {
            arg0.vertexAttribIPointer(arg1 >>> 0, arg2, arg3 >>> 0, arg4, arg5);
        },
        __wbg_vertexAttribPointer_00b9f17c7c37643e: function(arg0, arg1, arg2, arg3, arg4, arg5, arg6) {
            arg0.vertexAttribPointer(arg1 >>> 0, arg2, arg3 >>> 0, arg4 !== 0, arg5, arg6);
        },
        __wbg_vertexAttribPointer_40a7271207736a85: function(arg0, arg1, arg2, arg3, arg4, arg5, arg6) {
            arg0.vertexAttribPointer(arg1 >>> 0, arg2, arg3 >>> 0, arg4 !== 0, arg5, arg6);
        },
        __wbg_videoHeight_25afdcce125cb67f: function(arg0) {
            const ret = arg0.videoHeight;
            return ret;
        },
        __wbg_videoWidth_a9f6134cf31b4894: function(arg0) {
            const ret = arg0.videoWidth;
            return ret;
        },
        __wbg_viewport_5f258767db23cc99: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.viewport(arg1, arg2, arg3, arg4);
        },
        __wbg_viewport_fe0597169ee713fe: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.viewport(arg1, arg2, arg3, arg4);
        },
        __wbg_warn_89e632e51aaa7502: function(arg0) {
            console.warn(arg0);
        },
        __wbg_width_29e989639db86d0c: function(arg0) {
            const ret = arg0.width;
            return ret;
        },
        __wbg_width_6ae61fc4b0a15f1c: function(arg0) {
            const ret = arg0.width;
            return ret;
        },
        __wbg_width_6b854cf8da3aeb47: function(arg0) {
            const ret = arg0.width;
            return ret;
        },
        __wbg_window_9c17850b5e99c0ab: function() { return handleError(function () {
            const ret = window.window;
            return ret;
        }, arguments); },
        __wbg_x_bdc2f0bd37149938: function(arg0) {
            const ret = arg0.x;
            return ret;
        },
        __wbg_y_5d53694f65948821: function(arg0) {
            const ret = arg0.y;
            return ret;
        },
        __wbindgen_cast_0000000000000001: function(arg0, arg1) {
            // Cast intrinsic for `Closure(Closure { owned: true, function: Function { arguments: [Externref], shim_idx: 136, ret: Unit, inner_ret: Some(Unit) }, mutable: true }) -> Externref`.
            const ret = makeMutClosure(arg0, arg1, wasm_bindgen__convert__closures_____invoke__h6e08faa3963ece35);
            return ret;
        },
        __wbindgen_cast_0000000000000002: function(arg0, arg1) {
            // Cast intrinsic for `Closure(Closure { owned: true, function: Function { arguments: [F64], shim_idx: 102, ret: Unit, inner_ret: Some(Unit) }, mutable: true }) -> Externref`.
            const ret = makeMutClosure(arg0, arg1, wasm_bindgen__convert__closures_____invoke__h585079869faa3efb);
            return ret;
        },
        __wbindgen_cast_0000000000000003: function(arg0, arg1) {
            // Cast intrinsic for `Closure(Closure { owned: true, function: Function { arguments: [NamedExternref("BeforeUnloadEvent")], shim_idx: 96, ret: Unit, inner_ret: Some(Unit) }, mutable: true }) -> Externref`.
            const ret = makeMutClosure(arg0, arg1, wasm_bindgen__convert__closures_____invoke__h02b2901519431638);
            return ret;
        },
        __wbindgen_cast_0000000000000004: function(arg0, arg1) {
            // Cast intrinsic for `Closure(Closure { owned: true, function: Function { arguments: [NamedExternref("Event")], shim_idx: 96, ret: Unit, inner_ret: Some(Unit) }, mutable: true }) -> Externref`.
            const ret = makeMutClosure(arg0, arg1, wasm_bindgen__convert__closures_____invoke__h02b2901519431638_3);
            return ret;
        },
        __wbindgen_cast_0000000000000005: function(arg0, arg1) {
            // Cast intrinsic for `Closure(Closure { owned: true, function: Function { arguments: [NamedExternref("FocusEvent")], shim_idx: 96, ret: Unit, inner_ret: Some(Unit) }, mutable: true }) -> Externref`.
            const ret = makeMutClosure(arg0, arg1, wasm_bindgen__convert__closures_____invoke__h02b2901519431638_4);
            return ret;
        },
        __wbindgen_cast_0000000000000006: function(arg0, arg1) {
            // Cast intrinsic for `Closure(Closure { owned: true, function: Function { arguments: [NamedExternref("KeyboardEvent")], shim_idx: 96, ret: Unit, inner_ret: Some(Unit) }, mutable: true }) -> Externref`.
            const ret = makeMutClosure(arg0, arg1, wasm_bindgen__convert__closures_____invoke__h02b2901519431638_5);
            return ret;
        },
        __wbindgen_cast_0000000000000007: function(arg0, arg1) {
            // Cast intrinsic for `Closure(Closure { owned: true, function: Function { arguments: [NamedExternref("MediaQueryListEvent")], shim_idx: 96, ret: Unit, inner_ret: Some(Unit) }, mutable: true }) -> Externref`.
            const ret = makeMutClosure(arg0, arg1, wasm_bindgen__convert__closures_____invoke__h02b2901519431638_6);
            return ret;
        },
        __wbindgen_cast_0000000000000008: function(arg0, arg1) {
            // Cast intrinsic for `Closure(Closure { owned: true, function: Function { arguments: [NamedExternref("MessageEvent")], shim_idx: 96, ret: Unit, inner_ret: Some(Unit) }, mutable: true }) -> Externref`.
            const ret = makeMutClosure(arg0, arg1, wasm_bindgen__convert__closures_____invoke__h02b2901519431638_7);
            return ret;
        },
        __wbindgen_cast_0000000000000009: function(arg0, arg1) {
            // Cast intrinsic for `Closure(Closure { owned: true, function: Function { arguments: [NamedExternref("MouseEvent")], shim_idx: 96, ret: Unit, inner_ret: Some(Unit) }, mutable: true }) -> Externref`.
            const ret = makeMutClosure(arg0, arg1, wasm_bindgen__convert__closures_____invoke__h02b2901519431638_8);
            return ret;
        },
        __wbindgen_cast_000000000000000a: function(arg0, arg1) {
            // Cast intrinsic for `Closure(Closure { owned: true, function: Function { arguments: [NamedExternref("PointerEvent")], shim_idx: 96, ret: Unit, inner_ret: Some(Unit) }, mutable: true }) -> Externref`.
            const ret = makeMutClosure(arg0, arg1, wasm_bindgen__convert__closures_____invoke__h02b2901519431638_9);
            return ret;
        },
        __wbindgen_cast_000000000000000b: function(arg0, arg1) {
            // Cast intrinsic for `Closure(Closure { owned: true, function: Function { arguments: [NamedExternref("WheelEvent")], shim_idx: 96, ret: Unit, inner_ret: Some(Unit) }, mutable: true }) -> Externref`.
            const ret = makeMutClosure(arg0, arg1, wasm_bindgen__convert__closures_____invoke__h02b2901519431638_10);
            return ret;
        },
        __wbindgen_cast_000000000000000c: function(arg0, arg1) {
            // Cast intrinsic for `Closure(Closure { owned: true, function: Function { arguments: [], shim_idx: 104, ret: Unit, inner_ret: Some(Unit) }, mutable: true }) -> Externref`.
            const ret = makeMutClosure(arg0, arg1, wasm_bindgen__convert__closures_____invoke__hea2ef47cebb1cba9);
            return ret;
        },
        __wbindgen_cast_000000000000000d: function(arg0) {
            // Cast intrinsic for `F64 -> Externref`.
            const ret = arg0;
            return ret;
        },
        __wbindgen_cast_000000000000000e: function(arg0, arg1) {
            // Cast intrinsic for `Ref(String) -> Externref`.
            const ret = getStringFromWasm0(arg0, arg1);
            return ret;
        },
        __wbindgen_init_externref_table: function() {
            const table = wasm.__wbindgen_externrefs;
            const offset = table.grow(4);
            table.set(0, undefined);
            table.set(offset + 0, undefined);
            table.set(offset + 1, null);
            table.set(offset + 2, true);
            table.set(offset + 3, false);
        },
    };
    return {
        __proto__: null,
        "./onion_bg.js": import0,
    };
}

function wasm_bindgen__convert__closures_____invoke__hea2ef47cebb1cba9(arg0, arg1) {
    wasm.wasm_bindgen__convert__closures_____invoke__hea2ef47cebb1cba9(arg0, arg1);
}

function wasm_bindgen__convert__closures_____invoke__h6e08faa3963ece35(arg0, arg1, arg2) {
    wasm.wasm_bindgen__convert__closures_____invoke__h6e08faa3963ece35(arg0, arg1, arg2);
}

function wasm_bindgen__convert__closures_____invoke__h02b2901519431638(arg0, arg1, arg2) {
    wasm.wasm_bindgen__convert__closures_____invoke__h02b2901519431638(arg0, arg1, arg2);
}

function wasm_bindgen__convert__closures_____invoke__h02b2901519431638_3(arg0, arg1, arg2) {
    wasm.wasm_bindgen__convert__closures_____invoke__h02b2901519431638_3(arg0, arg1, arg2);
}

function wasm_bindgen__convert__closures_____invoke__h02b2901519431638_4(arg0, arg1, arg2) {
    wasm.wasm_bindgen__convert__closures_____invoke__h02b2901519431638_4(arg0, arg1, arg2);
}

function wasm_bindgen__convert__closures_____invoke__h02b2901519431638_5(arg0, arg1, arg2) {
    wasm.wasm_bindgen__convert__closures_____invoke__h02b2901519431638_5(arg0, arg1, arg2);
}

function wasm_bindgen__convert__closures_____invoke__h02b2901519431638_6(arg0, arg1, arg2) {
    wasm.wasm_bindgen__convert__closures_____invoke__h02b2901519431638_6(arg0, arg1, arg2);
}

function wasm_bindgen__convert__closures_____invoke__h02b2901519431638_7(arg0, arg1, arg2) {
    wasm.wasm_bindgen__convert__closures_____invoke__h02b2901519431638_7(arg0, arg1, arg2);
}

function wasm_bindgen__convert__closures_____invoke__h02b2901519431638_8(arg0, arg1, arg2) {
    wasm.wasm_bindgen__convert__closures_____invoke__h02b2901519431638_8(arg0, arg1, arg2);
}

function wasm_bindgen__convert__closures_____invoke__h02b2901519431638_9(arg0, arg1, arg2) {
    wasm.wasm_bindgen__convert__closures_____invoke__h02b2901519431638_9(arg0, arg1, arg2);
}

function wasm_bindgen__convert__closures_____invoke__h02b2901519431638_10(arg0, arg1, arg2) {
    wasm.wasm_bindgen__convert__closures_____invoke__h02b2901519431638_10(arg0, arg1, arg2);
}

function wasm_bindgen__convert__closures_____invoke__h585079869faa3efb(arg0, arg1, arg2) {
    wasm.wasm_bindgen__convert__closures_____invoke__h585079869faa3efb(arg0, arg1, arg2);
}

function addToExternrefTable0(obj) {
    const idx = wasm.__externref_table_alloc();
    wasm.__wbindgen_externrefs.set(idx, obj);
    return idx;
}

const CLOSURE_DTORS = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(state => wasm.__wbindgen_destroy_closure(state.a, state.b));

function debugString(val) {
    // primitive types
    const type = typeof val;
    if (type == 'number' || type == 'boolean' || val == null) {
        return  `${val}`;
    }
    if (type == 'string') {
        return `"${val}"`;
    }
    if (type == 'symbol') {
        const description = val.description;
        if (description == null) {
            return 'Symbol';
        } else {
            return `Symbol(${description})`;
        }
    }
    if (type == 'function') {
        const name = val.name;
        if (typeof name == 'string' && name.length > 0) {
            return `Function(${name})`;
        } else {
            return 'Function';
        }
    }
    // objects
    if (Array.isArray(val)) {
        const length = val.length;
        let debug = '[';
        if (length > 0) {
            debug += debugString(val[0]);
        }
        for(let i = 1; i < length; i++) {
            debug += ', ' + debugString(val[i]);
        }
        debug += ']';
        return debug;
    }
    // Test for built-in
    const builtInMatches = /\[object ([^\]]+)\]/.exec(toString.call(val));
    let className;
    if (builtInMatches && builtInMatches.length > 1) {
        className = builtInMatches[1];
    } else {
        // Failed to match the standard '[object ClassName]'
        return toString.call(val);
    }
    if (className == 'Object') {
        // we're a user defined class or Object
        // JSON.stringify avoids problems with cycles, and is generally much
        // easier than looping through ownProperties of `val`.
        try {
            return 'Object(' + JSON.stringify(val) + ')';
        } catch (_) {
            return 'Object';
        }
    }
    // errors
    if (val instanceof Error) {
        return `${val.name}: ${val.message}\n${val.stack}`;
    }
    // TODO we could test for more things here, like `Set`s and `Map`s.
    return className;
}

function getArrayF32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getFloat32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

function getArrayI32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getInt32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

function getArrayU32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

let cachedDataViewMemory0 = null;
function getDataViewMemory0() {
    if (cachedDataViewMemory0 === null || cachedDataViewMemory0.buffer.detached === true || (cachedDataViewMemory0.buffer.detached === undefined && cachedDataViewMemory0.buffer !== wasm.memory.buffer)) {
        cachedDataViewMemory0 = new DataView(wasm.memory.buffer);
    }
    return cachedDataViewMemory0;
}

let cachedFloat32ArrayMemory0 = null;
function getFloat32ArrayMemory0() {
    if (cachedFloat32ArrayMemory0 === null || cachedFloat32ArrayMemory0.byteLength === 0) {
        cachedFloat32ArrayMemory0 = new Float32Array(wasm.memory.buffer);
    }
    return cachedFloat32ArrayMemory0;
}

let cachedInt32ArrayMemory0 = null;
function getInt32ArrayMemory0() {
    if (cachedInt32ArrayMemory0 === null || cachedInt32ArrayMemory0.byteLength === 0) {
        cachedInt32ArrayMemory0 = new Int32Array(wasm.memory.buffer);
    }
    return cachedInt32ArrayMemory0;
}

function getStringFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return decodeText(ptr, len);
}

let cachedUint32ArrayMemory0 = null;
function getUint32ArrayMemory0() {
    if (cachedUint32ArrayMemory0 === null || cachedUint32ArrayMemory0.byteLength === 0) {
        cachedUint32ArrayMemory0 = new Uint32Array(wasm.memory.buffer);
    }
    return cachedUint32ArrayMemory0;
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function handleError(f, args) {
    try {
        return f.apply(this, args);
    } catch (e) {
        const idx = addToExternrefTable0(e);
        wasm.__wbindgen_exn_store(idx);
    }
}

function isLikeNone(x) {
    return x === undefined || x === null;
}

function makeMutClosure(arg0, arg1, f) {
    const state = { a: arg0, b: arg1, cnt: 1 };
    const real = (...args) => {

        // First up with a closure we increment the internal reference
        // count. This ensures that the Rust closure environment won't
        // be deallocated while we're invoking it.
        state.cnt++;
        const a = state.a;
        state.a = 0;
        try {
            return f(a, state.b, ...args);
        } finally {
            state.a = a;
            real._wbg_cb_unref();
        }
    };
    real._wbg_cb_unref = () => {
        if (--state.cnt === 0) {
            wasm.__wbindgen_destroy_closure(state.a, state.b);
            state.a = 0;
            CLOSURE_DTORS.unregister(state);
        }
    };
    CLOSURE_DTORS.register(real, state, state);
    return real;
}

function passStringToWasm0(arg, malloc, realloc) {
    if (realloc === undefined) {
        const buf = cachedTextEncoder.encode(arg);
        const ptr = malloc(buf.length, 1) >>> 0;
        getUint8ArrayMemory0().subarray(ptr, ptr + buf.length).set(buf);
        WASM_VECTOR_LEN = buf.length;
        return ptr;
    }

    let len = arg.length;
    let ptr = malloc(len, 1) >>> 0;

    const mem = getUint8ArrayMemory0();

    let offset = 0;

    for (; offset < len; offset++) {
        const code = arg.charCodeAt(offset);
        if (code > 0x7F) break;
        mem[ptr + offset] = code;
    }
    if (offset !== len) {
        if (offset !== 0) {
            arg = arg.slice(offset);
        }
        ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
        const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
        const ret = cachedTextEncoder.encodeInto(arg, view);

        offset += ret.written;
        ptr = realloc(ptr, len, offset, 1) >>> 0;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
cachedTextDecoder.decode();
const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

const cachedTextEncoder = new TextEncoder();

if (!('encodeInto' in cachedTextEncoder)) {
    cachedTextEncoder.encodeInto = function (arg, view) {
        const buf = cachedTextEncoder.encode(arg);
        view.set(buf);
        return {
            read: arg.length,
            written: buf.length
        };
    };
}

let WASM_VECTOR_LEN = 0;

let wasmModule, wasm;
function __wbg_finalize_init(instance, module) {
    wasm = instance.exports;
    wasmModule = module;
    cachedDataViewMemory0 = null;
    cachedFloat32ArrayMemory0 = null;
    cachedInt32ArrayMemory0 = null;
    cachedUint32ArrayMemory0 = null;
    cachedUint8ArrayMemory0 = null;
    wasm.__wbindgen_start();
    return wasm;
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = module.ok && expectedResponseType(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else { throw e; }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }

    function expectedResponseType(type) {
        switch (type) {
            case 'basic': case 'cors': case 'default': return true;
        }
        return false;
    }
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (module !== undefined) {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (module_or_path !== undefined) {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (module_or_path === undefined) {
        module_or_path = new URL('onion_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync, __wbg_init as default };
